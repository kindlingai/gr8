// Freezes what the S2 service calls on a model module. Two ranks (a head and a worker over a socket pair) run the
// real Service, Driver, scheduler, pools and event gate over a fake module that logs every call it gets, with its
// arguments and what it returns, through scenarios that reach every hook: burst admissions, prefill quanta and packs,
// growth, prefix reuse, cancels, a shrink, a resize, DFlash2 and parks where the shape has them, and a tight pool where an
// admission fits only on the pages a cancelled reused run gives back, and a cancel of a session still in its prefill
// (the head's resp files' last lines join that transcript). A command meant to land mid-run is written from
// inside a head step. The step then waits for its notice in the head's trace, so the boundary the notice names, and
// every call after it, is the same on every run. The ranks' steps meet as a fabric's would.
// The service's plain pool calls (reserve, fits, room, count) are logged by observers: the head's control thread (Rust)
// reports its own, and a worker's bind calls a RunPool that hides the real one's methods, which every header after
// pool.h names in its place. Not logged, because how many there are depends on the clock: the pool reads of the head's
// status (free pages, room, stats, runs, tiers) and share polls that find nothing. Each scenario runs twice, its module
// calls and pool policy as std::function hooks and as tables (ModuleOps, PoolOps). Each rank's transcript is hashed
// (FNV-1a). The hashes were recorded from the C++ service, in each shape.
//   --dump   print every transcript
// The generator is mt19937_64 with modulo ranges, and no call takes two draws as arguments. Host only.
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <dirent.h>
#include <unistd.h>
#define RunPool RunPoolBase
#include "../pool.h"
#undef RunPool
namespace g6s2 {
// the test's observer on the pool: each plain call the service makes, with what it returned, to `see`
class RunPool : public RunPoolBase {
public:
    using RunPoolBase::RunPoolBase;
    using RunPoolBase::reserve;
    std::function<void(const std::string&)> see;
    Run reserve(uint64_t rows, uint64_t worst, uint32_t r = 0);
    bool fits(uint64_t rows, uint64_t worst, uint32_t r = 0) const;
    uint32_t count(const std::vector<int>& prompt, int64_t span, const g6rt::CacheNamespace& ns = {});
};
} // namespace g6s2
#include "../service.h"

using namespace g6s2;
namespace {
template <class Tag, typename Tag::type M> struct Rob { friend typename Tag::type get(Tag) { return M; } };
struct ControlOf { using type = ControlHandle Service::*; friend type get(ControlOf); };
template struct Rob<ControlOf, &Service::ctl_>;

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint64_t U(uint64_t lo, uint64_t hi) { return lo + g() % (hi - lo + 1); }
};
uint64_t fnv(const void* p, size_t n, uint64_t h = 14695981039346656037ull) {
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(p)[i]) * 1099511628211ull;
    return h;
}
uint64_t fnv_ints(const std::vector<int>& v) { return fnv(v.data(), v.size() * sizeof(int)); }
uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
std::string u(uint64_t x) { return std::to_string(x); }
std::string i(int64_t x) { return std::to_string(x); }
std::string str(const Key& k) { return "k" + u(k.epoch) + "." + u(k.slot) + "." + u(k.generation); }
std::string str(const Run& r) { return "run" + u(r.first_page) + "+" + u(r.pages) + "v" + u(r.table_version) + "#" + u(r.list_hash); }
std::string str(const Request& r) {
    return "req" + u(r.id) + " p" + u(r.prompt_tokens) + " m" + u(r.max_tokens) + " r" + u(r.r_rows) + " st" + u(r.steering.enabled);
}
std::string str(const Grant& g) {
    return str(g.key) + " " + str(g.req) + " " + str(g.run) + " reuse " + u(g.reuse) + " restored " + u(g.restored) + " counted " + u(g.counted);
}
std::string str(const Slot& s) {
    return str(s.key) + " " + str(s.req) + " " + str(s.run) + " ph" + u(unsigned(s.phase)) + " pf" + u(s.prefilled) + " em" + u(s.emitted) +
           " in" + u(s.inserted) + " df" + u(s.deferred) + " pk" + u(s.packable) + u(s.pack_tail) + " parked" + u(s.parked);
}
std::string str(const Payload& p) {
    return "prompt " + u(p.prompt.size()) + "#" + u(fnv_ints(p.prompt)) + " chunk " + i(p.chunk) + " dp " + u(p.decode_prompt) +
           " images " + u(p.images.size());
}
std::string str(const g6rt::CacheNamespace& ns) { return "ns#" + u(fnv(ns.data(), ns.size())); }
std::string str(const g6rt::PrefixReuse& st) {
    return "hist " + u(st.history.size()) + "#" + u(fnv_ints(st.history)) + " valid " + i(st.valid) + " main " + i(st.main_end) +
           " draft " + i(st.draft_end) + " written " + i(st.written_end) + " pf " + i(st.pf_end) + " dirty " + i(st.dirty_end) + " " +
           str(st.cache_namespace);
}
std::string str(const PoolHit& h) {
    return "hit skip " + i(h.skip) + " lcp " + i(h.lcp) + " why " + h.why + " age " + u(h.age) + " stored " + u(h.stored) + " tier " +
           u(h.tier) + " slot " + i(h.slot);
}
std::string str(const Resize& r) { return "resize t" + u(r.tier) + " segs " + u(r.segs) + " slots " + u(r.slots) + " txn " + u(r.txn); }
std::string str(const Shrink& s) { return str(s.key) + " r" + u(s.r_rows) + " " + str(s.run); }
std::string str(const Sampling& s) {
    return "smp " + u(bits(s.inv_temp)) + " " + u(bits(s.top_p)) + " " + u(s.seed) + " " + i(s.lp) + " " + i(s.plp);
}
std::string ints(const std::vector<int>& v) {
    std::string s;
    for (size_t k = 0; k < v.size(); ++k) s += (k ? "," : "") + i(v[k]);
    return s;
}
std::string str(const Launch& l) {
    const s2::Step& t = l.step;
    std::string s = "kind " + i(int(t.kind)) + " M " + i(t.M()) + " ev " + i(int(t.event)) + " cls " + i(t.cls) + " round " + i(t.round) +
                    " dec " + u(t.scheduler_decode) + " sess " + ints(t.sess) + " row0 " + ints(t.row0) + " rows " + ints(t.rows) + " ctx " +
                    ints(t.ctx) + " evs " + ints(t.ev) + " r";
    for (const auto& r : t.r) s += " " + i(r.sess) + ":" + i(r.tok) + "@" + i(r.pos) + "/" + i(r.spos) + "t" + i(r.tgt);
    s += " seq " + u(l.sequence) + " solo " + i(l.solo) + " context " + i(l.context) + " requests";
    for (uint64_t x : l.requests) s += " " + u(x);
    s += " slots " + ints(l.req_slots) + " events " + ints(l.events) + " event " + i(int(l.event)) + " buckets " + ints(l.buckets) +
         " src " + ints(l.source_rows) + " bucket " + i(l.bucket) + " key " + l.graph_key;
    for (const auto& x : l.samples) s += " " + str(x);
    for (float x : l.steering_scales) s += " sc" + u(bits(x));
    s += " sslots " + ints(l.steering_slots);
    return s;
}
std::string str(const s2::DflReq& r) {
    return "dfl sess " + i(r.sess) + " rows " + i(r.rows) + " pos0 " + i(r.pos0) + " tok0 " + i(r.tok0) + " lo " + i(r.lo) + " q " +
           u(r.query) + " phase " + i(r.phase);
}
} // namespace
Run g6s2::RunPool::reserve(uint64_t rows, uint64_t worst, uint32_t r) {
    const Run x = RunPoolBase::reserve(rows, worst, r);
    if (see) see("reserve " + u(rows) + " " + u(worst) + " r" + u(r) + " -> " + str(x));
    return x;
}
bool g6s2::RunPool::fits(uint64_t rows, uint64_t worst, uint32_t r) const {
    const bool f = RunPoolBase::fits(rows, worst, r);
    if (see) see("fits " + u(rows) + " " + u(worst) + " r" + u(r) + " -> " + u(f));
    return f;
}
uint32_t g6s2::RunPool::count(const std::vector<int>& prompt, int64_t span, const g6rt::CacheNamespace& ns) {
    const uint32_t n = RunPoolBase::count(prompt, span, ns);
    if (see) see("count " + u(prompt.size()) + "#" + u(fnv_ints(prompt)) + " span " + i(span) + " " + str(ns) + " -> " + u(n));
    return n;
}
namespace {
// the head's control thread's plain pool calls, in the RunPool observer's words
void seen(void* ctx, const g6svc_pool_call* c) {
    auto& p = *static_cast<RunPool*>(ctx);
    if (!p.see) return;
    if (c->kind == 0) p.see("room -> " + u(c->out));
    else if (c->kind == 1) p.see("fits " + u(c->rows) + " " + u(c->worst) + " r" + u(c->r) + " -> " + u(c->out));
    else if (c->kind == 2)
        p.see("reserve " + u(c->rows) + " " + u(c->worst) + " r" + u(c->r) + " -> " + str(*sch::c<Run>(&c->run)));
    else {
        g6rt::CacheNamespace ns;
        std::memcpy(ns.data(), c->ns, ns.size());
        p.see("count " + u(c->n) + "#" + u(fnv(c->prompt, c->n * sizeof(int))) + " span " + i(c->span) + " " + str(ns) + " -> " +
              u(c->out));
    }
}
uint64_t mix(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ (c + 7) * 0x94D049BB133111EBull;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return x;
}
std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream o; o << f.rdbuf(); return o.str();
}
size_t count(const std::string& s, const std::string& w) {
    size_t n = 0;
    for (size_t at = s.find(w); at != std::string::npos; at = s.find(w, at + 1)) ++n;
    return n;
}
// removes a directory and every file in it
void remove_dir(const std::string& d) {
    if (DIR* x = ::opendir(d.c_str())) {
        while (const dirent* e = ::readdir(x))
            if (std::strcmp(e->d_name, ".") && std::strcmp(e->d_name, "..")) ::unlink((d + "/" + e->d_name).c_str());
        ::closedir(x);
    }
    ::rmdir(d.c_str());
}
std::string fresh_dir() {
    char t[] = "/tmp/module_calls_XXXXXX";
    require(::mkdtemp(t) != nullptr, "mkdtemp");
    return t;
}

// A request a scenario sends: its id, prompt and header
struct Req {
    uint64_t id = 0; std::vector<int> prompt; std::string head = "n_gen 8 temp 0 seed 1 min 0 stop -";
    std::string bytes() const { return head + "\n" + ints(prompt) + "\n"; }
};
// What the head does inside its launch number `at`: send these, then wait for one notice each
struct Trigger { long at = 0; std::vector<uint64_t> admit, cancel; bool resize = false; };
// reqs: written before the phase starts. late: sent by the triggers
struct Stage { std::vector<Req> reqs, late; std::vector<Trigger> triggers; };
struct Scenario {
    std::string name;
    Config cfg;
    uint32_t prefill_min = 0, pages = 256, place = 256, below = 256 * 64;
    bool window = false;   // a window pool (kvman-rs Pool::window): runs past R, r_rows and make_room
    bool reuse = false, park = false, dfl = false, seq = false, drained = false, grow = true, kstop_log = false;
    bool ends = false;   // the head's transcript ends with each request's last two resp lines (its "m" and "end")
    std::vector<Stage> phases;
};

// the fabric: a rank's step waits for the other rank's step of the same number
struct Lockstep {
    std::mutex mu; std::condition_variable cv; uint64_t n[2] = {0, 0};
    void step(unsigned rank) {
        std::unique_lock<std::mutex> l(mu);
        ++n[rank]; cv.notify_all();
        cv.wait_for(l, std::chrono::seconds(20), [&] { return n[1 - rank] >= n[rank]; });
    }
};

struct Fake {
    unsigned rank; const Scenario& sc; Lockstep& ls;
    std::mutex mu; std::vector<std::string> lines;
    std::unique_ptr<RunPool> pool;
    long launches = 0, drains = 0;
    std::function<void(long)> on_launch;              // head: the triggers
    std::atomic<bool> resize_due{false};
    uint64_t resizes = 0;
    std::array<uint32_t, MAX_SESSIONS> prompt{};      // each slot's bound prompt length
    // Config::park, as DS41's adapter keeps it: a finished chat's tokens on its slot, and the slot a take claimed
    struct Parked { bool on = false; uint32_t pos = 0; std::vector<int> tokens; g6rt::CacheNamespace ns{}; };
    std::array<Parked, MAX_SESSIONS> parked{};
    std::atomic<int> claimed{-1};
    Fake(unsigned r, const Scenario& s, Lockstep& l) : rank(r), sc(s), ls(l) {
        pool = std::make_unique<RunPool>(s.pages, s.place, s.below);
        pool->see = [this](const std::string& x) { tx(x); };
        if (s.window) { pool->tail(4, 0); pool->window(4, 0, 0); }
        if (s.reuse && !s.park) { g6rt::PrefixReuse rules; pool->keep(rules); }
    }
    void tx(const std::string& s) { std::lock_guard<std::mutex> l(mu); lines.push_back(s); }
    std::vector<int> ids(const Launch& l) const {
        std::vector<int> v;
        for (const auto& r : l.step.r)
            v.push_back(100 + int(mix(uint64_t(r.sess) + 17, uint64_t(r.pos) * 2 + (l.step.kind != s2::Step::MAIN), uint64_t(r.tok)) % 48));
        return v;
    }
    NativeHooks hooks() {
        NativeHooks h;
        RunPool& p = *pool;
        h.pool.pool = &p;
        h.pool.digest = [this, &p] {
            uint64_t d = p.digest();
            for (const Parked& k : parked) d = (d ^ (uint64_t(k.on) + (uint64_t(k.pos) << 1))) * 0x100000001B3ull;
            tx("digest -> " + u(d));
            return d;
        };
        h.pool.grow = sc.grow;
        if (sc.window) {
            h.pool.refusal = [this](uint64_t worst, uint32_t r) { tx("refusal " + u(worst) + " r" + u(r)); return std::string("no"); };
            h.pool.r_rows = [this, &p](const Request& r, uint64_t rows, uint64_t worst, const std::vector<int>* prompt) -> uint32_t {
                uint32_t out = 0;
                if (worst > sc.cfg.r2_rows) out = p.window_runs(0) < 1 && p.fits(rows, worst, 0) ? 0 : sc.cfg.r2_rows;
                tx("r_rows " + str(r) + " " + u(rows) + " " + u(worst) + " prompt " + (prompt ? u(prompt->size()) : "-") + " -> " + u(out));
                return out;
            };
            h.pool.make_room = [this, &p](const std::array<Slot, MAX_SESSIONS>& slots, const Request& r, uint64_t rows, uint64_t worst)
                -> std::optional<Shrink> {
                std::optional<Shrink> out;
                for (const Slot& s : slots)
                    if (s.phase == Phase::Decode && !s.req.r_rows && p.shrinkable(s.run, sc.cfg.r2_rows)) {
                        out = Shrink{s.key, sc.cfg.r2_rows, p.shrink(s.run, sc.cfg.r2_rows)};
                        break;
                    }
                tx("make_room " + str(r) + " " + u(rows) + " " + u(worst) + " -> " + (out ? str(*out) : "none"));
                return out;
            };
        }
        if (sc.reuse) {
            h.pool.count = true;
            if (!sc.park) {
                h.pool.retain = [this, &p](const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) {
                    tx("retain " + str(r) + " " + str(st) + " " + str(ns));
                    p.retain(r, st, ns);
                };
                h.pool.match = [this, &p](const std::vector<int>& pr, uint64_t rows, uint64_t worst, int limit, const g6rt::CacheNamespace& ns,
                                          uint32_t r) {
                    const PoolHit x = p.match(pr, rows, worst, limit, ns, r);
                    tx("match " + u(pr.size()) + "#" + u(fnv_ints(pr)) + " " + u(rows) + " " + u(worst) + " " + i(limit) + " " + str(ns) + " r" +
                       u(r) + " -> " + str(x));
                    return x;
                };
                h.pool.take = [this, &p](const PoolHit& x, uint64_t rows, uint64_t worst, uint32_t r) {
                    const Run y = p.take(x, rows, worst, r);
                    tx("take " + str(x) + " " + u(rows) + " " + u(worst) + " r" + u(r) + " -> " + str(y));
                    return y;
                };
            } else {
                h.pool.retain = [this, &p](const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) {
                    tx("retain " + str(r) + " " + str(st) + " " + str(ns));
                    p.release(r);
                };
                h.pool.park = [this, &p](unsigned slot, const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) {
                    tx("park " + u(slot) + " " + str(r) + " " + str(st) + " " + str(ns));
                    const uint32_t pos = st.history.empty() ? 0u : uint32_t(std::min<int>(st.main_end, int(st.history.size()) - 1));
                    {
                        std::lock_guard<std::mutex> l(mu);
                        Parked& k = parked[slot];
                        k.on = pos > 0; k.pos = pos; k.ns = ns;
                        k.tokens.assign(st.history.begin(), st.history.begin() + std::min<size_t>(pos, st.history.size()));
                    }
                    p.release(r);
                };
                h.pool.match = [this, &p](const std::vector<int>& pr, uint64_t rows, uint64_t worst, int limit, const g6rt::CacheNamespace& ns,
                                          uint32_t r) {
                    PoolHit x;
                    int q = -1;
                    {
                        std::lock_guard<std::mutex> l(mu);
                        for (unsigned s = 0; s < MAX_SESSIONS; ++s) {
                            const Parked& k = parked[s];
                            if (!k.on || k.ns != ns || k.pos > uint32_t(std::max(limit, 0)) ||
                                g6rt::PrefixReuse::common_prefix(k.tokens, pr) < int(k.pos)) continue;
                            if (q < 0 || k.pos > parked[size_t(q)].pos) q = int(s);
                        }
                        if (q >= 0 && p.fits(rows, worst, r)) { x.skip = int(parked[size_t(q)].pos); x.lcp = x.skip; x.slot = q; x.why = "parked"; }
                    }
                    tx("match " + u(pr.size()) + "#" + u(fnv_ints(pr)) + " " + u(rows) + " " + u(worst) + " " + i(limit) + " " + str(ns) + " r" +
                       u(r) + " -> " + str(x));
                    return x;
                };
                h.pool.take = [this, &p](const PoolHit& x, uint64_t rows, uint64_t worst, uint32_t r) {
                    claimed.store(x.slot);
                    const Run y = p.reserve(rows, worst, r);
                    tx("take " + str(x) + " " + u(rows) + " " + u(worst) + " r" + u(r) + " -> " + str(y));
                    return y;
                };
            }
        }
        h.bind_request = [this](unsigned slot, const g6rt::ServeReq& r) {
            tx("bind_request " + u(slot) + " prompt " + u(r.prompt.size()) + "#" + u(fnv_ints(r.prompt)) + " n_gen " + i(r.n_gen) + " lp " + i(r.lp) +
               " plp " + i(r.plp) + " images " + u(r.images.size()));
        };
        h.bind = [this](const Grant& g) {
            prompt[g.key.slot] = g.req.prompt_tokens;
            std::string park;
            if (sc.park) {
                std::lock_guard<std::mutex> l(mu);
                const int cs = claimed.exchange(-1);
                park = " claimed " + i(cs) + " parked " + u(parked[g.key.slot].on);
                parked[g.key.slot] = Parked{};
            }
            tx("bind " + str(g) + park);
        };
        h.grown = [this](unsigned slot, const Run& r) { tx("grown " + u(slot) + " " + str(r)); };
        h.retire = [this](const Slot& s) { tx("retire " + str(s)); };
        h.fence = [this] { tx("fence"); };
        h.before_work = [this] { tx("before_work"); };
        if (sc.drained)
            h.drained = [this] {
                const bool d = ++drains % 4 == 0;
                tx("drained -> " + u(d));
                return d;
            };
        h.flash_room = [this] { tx("flash_room -> 9"); return uint32_t(9); };
        h.resize = [this](const Resize& r) { tx(str(r)); };
        if (sc.window) h.shrink = [this](const Shrink& k, const Run& was, uint64_t length) {
            tx("shrink " + str(k) + " was " + str(was) + " length " + u(length));
        };
        if (rank == 0)
            h.share = [this](bool idle) -> std::optional<Resize> {
                if (!resize_due.exchange(false)) return std::nullopt;
                ++resizes;
                const Resize r{3, 10 + resizes, 100 * resizes, 7 + resizes};
                tx("share idle " + u(idle) + " -> " + str(r));
                return r;
            };
        h.dispatch.bucket = [this](bool draft, int ctx) {
            const int b = ctx <= 256 ? 0 : ctx <= 1024 ? 1 : 2;
            tx("bucket " + u(draft) + " " + i(ctx) + " -> " + i(b));
            return b;
        };
        h.dispatch.mode = [this](int old, int next) { tx("mode " + i(old) + " " + i(next)); };
        h.dispatch.launch = [this](const Launch& l) {
            ls.step(rank);
            const auto v = ids(l);
            tx("launch " + str(l) + " -> " + ints(v));
            ++launches;
            if (on_launch) on_launch(launches);
            return v;
        };
        h.dispatch.logprob = [this](int row, bool target, int k) {
            g6rt::LpRow r;
            r.lp = -0.125f * float(row + 1); r.rank = row % 3; r.n = std::min(k, 3);
            for (int j = 0; j < r.n; ++j) { r.id[j] = 100 + j; r.tlp[j] = -0.25f * float(j); }
            tx("logprob " + i(row) + " " + u(target) + " " + i(k));
            return r;
        };
        h.dispatch.draft_lp = [this](int row) {
            tx("draft_lp " + i(row));
            return -0.0625f * float(row % 5 + 1);
        };
        if (sc.seq) {
            h.dispatch.prefill_seq = [this](const Slot& s, const Payload& p, uint32_t first, uint32_t rows, uint64_t seq) {
                ls.step(rank);
                tx("prefill_seq " + str(s) + " " + str(p) + " " + u(first) + "+" + u(rows) + " seq " + u(seq));
            };
            h.dispatch.prefill_pack_seq = [this](const std::vector<PrefillPart>& parts, uint64_t seq) {
                ls.step(rank);
                std::string s = "prefill_pack_seq seq " + u(seq);
                for (const auto& x : parts) s += " [" + str(*x.slot) + " " + str(*x.payload) + " " + u(x.first) + "+" + u(x.rows) + "]";
                tx(s);
            };
        } else
            h.dispatch.prefill = [this](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
                ls.step(rank);
                tx("prefill " + str(s) + " " + str(p) + " " + u(first) + "+" + u(rows));
            };
        h.kstop.log = sc.kstop_log;
        if (sc.dfl) {
            h.dfl_kv = 7;
            h.dispatch.dfl_ok = [this](int q) { tx("dfl_ok " + i(q) + " -> 1"); return true; };
            h.dispatch.dfl_start = [this](int q) {
                const int at = int(prompt[size_t(q)]) - 1;
                tx("dfl_start " + i(q) + " -> " + i(at));
                return at;
            };
            h.dispatch.dfl = [this](const s2::DflReq& r, const Sampling& smp) {
                s2::DflDrafts o{};
                for (int j = 0; j < s2::DFL_B; ++j) {
                    o.tok[j] = 100 + int(mix(uint64_t(r.sess) + 17, uint64_t(r.pos0 + r.rows + j) * 2, uint64_t(r.tok0)) % 48);
                    o.lp[j] = -0.125f * float(j + 1);
                }
                tx(str(r) + " " + str(smp) + " -> " + i(o.tok[0]) + ".." + i(o.tok[s2::DFL_B - 1]));
                return o;
            };
        }
        return h;
    }
};

// the service's hooks with the module's calls and its pool policy moved into tables (ModuleOps, PoolOps) over `all`, as
// an adapter hands them over
NativeHooks as_table(const NativeHooks& all, FnModule<NativeHooks>& m, FnPool<PoolHooks>& p) {
    NativeHooks h;
    h.pool.pool = all.pool.pool; h.pool.ops = p.ops();
    h.share = all.share; h.flash = all.flash;
    h.dfl_kv = all.dfl_kv; h.dfl_spec = all.dfl_spec; h.dsel = all.dsel; h.kstop = all.kstop; h.dfl_reuse = all.dfl_reuse; h.ngram = all.ngram;
    h.module = m.ops();
    return h;
}
// runs one scenario and returns both ranks' transcripts. table: the service gets the module's calls as a table
std::array<std::vector<std::string>, 2> run(const Scenario& sc, bool table, std::string& error) {
    int sp[2];
    require(!::socketpair(AF_UNIX, SOCK_STREAM, 0, sp), "socketpair");
    ControlLinks head_links, worker_links;
    head_links.links.push_back(std::make_unique<ControlIO>(sp[0]));
    worker_links.links.push_back(std::make_unique<ControlIO>(sp[1]));
    for (auto* l : {&head_links, &worker_links}) l->links[0]->timeout_seconds(20);
    BootstrapConfig b[2];
    for (unsigned r = 0; r < 2; ++r) {
        b[r].scheduler = sc.cfg; b[r].ranks = 2; b[r].rank = r; b[r].head_ip = "127.0.0.1"; b[r].port = 1;
        b[r].runtime_signature = "module_calls_test"; b[r].directory = fresh_dir(); b[r].prefill_min = sc.prefill_min;
    }
    const std::string dir = b[0].directory, trace = knob::s2_trace().value_or("") + ".r0.json";
    Lockstep ls;
    Fake f0(0, sc, ls), f1(1, sc, ls);
    unsigned next_cmd = 0;
    auto send_req = [&](const Req& q) { atomic_file(dir + "/req." + u(q.id), q.bytes()); };
    auto admit = [&](const Req& q) {
        atomic_file(dir + "/cmd." + u(next_cmd++), "ADMIT " + u(sc.cfg.epoch) + " " + u(q.id) + " " + u(payload_hash(q.bytes())) + "\n");
    };
    auto notices = [&] { return count(slurp(trace), "{\"name\":\"notice\""); };
    std::map<uint64_t, const Req*> by_id;
    for (const auto& ph : sc.phases) for (const auto* l : {&ph.reqs, &ph.late}) for (const auto& q : *l) by_id[q.id] = &q;
    size_t phase = 0;
    // a trigger fires at its launch, or at the first after it with no event in flight (each commit is traced by this
    // thread, so the count at a launch does not depend on timing)
    std::set<const Trigger*> fired;
    f0.on_launch = [&](long n) {
        for (const auto& t : sc.phases[phase].triggers) {
            if (t.at > n || fired.count(&t)) continue;
            const std::string tr = slurp(trace);
            const size_t before = count(tr, "{\"name\":\"notice\"");
            if (count(tr, "{\"name\":\"commit\"") != before) continue;
            fired.insert(&t);
            size_t want = before;
            for (uint64_t id : t.cancel) { atomic_file(dir + "/cmd." + u(next_cmd++), "CANCEL " + u(sc.cfg.epoch) + " " + u(id) + " 0\n"); ++want; }
            for (uint64_t id : t.admit) { send_req(*by_id.at(id)); admit(*by_id.at(id)); ++want; }
            if (t.resize) { f0.resize_due = true; ++want; }
            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (notices() < want && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            require(notices() >= want, "a trigger's notice did not come");
            f0.tx("trigger " + i(t.at) + " at launch " + i(n));
        }
    };
    for (const auto& q : sc.phases[0].reqs) { send_req(q); admit(q); }
    std::exception_ptr err[2];
    std::unique_ptr<Service> s[2];
    NativeHooks all[2] = {f0.hooks(), f1.hooks()};
    std::unique_ptr<FnModule<NativeHooks>> mods[2];
    std::unique_ptr<FnPool<PoolHooks>> pools[2];
    try {
        for (unsigned r = 0; r < 2; ++r) {
            if (table) {
                mods[r] = std::make_unique<FnModule<NativeHooks>>(all[r]);
                pools[r] = std::make_unique<FnPool<PoolHooks>>(all[r].pool);
            }
            s[r] = std::make_unique<Service>(b[r], r ? worker_links : head_links, table ? as_table(all[r], *mods[r], *pools[r]) : all[r], 1);
            g6svc_control_observe(((*s[r]).*get(ControlOf{})).get(), seen, (r ? f1 : f0).pool.get());
        }
    } catch (const std::exception& e) { error = sc.name + ": " + e.what(); return {}; }
    s[0]->quit_linger_ms = s[1]->quit_linger_ms = 300;
    std::thread t0([&] { try { s[0]->run(); } catch (...) { err[0] = std::current_exception(); } });
    std::thread t1([&] { try { s[1]->run(); } catch (...) { err[1] = std::current_exception(); } });
    auto failed = [&] { return err[0] || err[1]; };
    // every request so far answered, and both ranks idle
    auto settle = [&](size_t upto) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        auto done = [&] {
            for (size_t k = 0; k <= upto; ++k)
                for (const auto* l : {&sc.phases[k].reqs, &sc.phases[k].late})
                    for (const auto& q : *l)
                        if (slurp(dir + "/resp." + u(q.id)).find("\nend ") == std::string::npos) return false;
            for (auto& x : s)
                if (g6svc_control_resident(((*x).*get(ControlOf{})).get())) return false;
            return true;
        };
        while (!done() && !failed() && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return done();
    };
    bool ok = settle(0);
    // a later phase's ADMITs, the first written last: the head reads cmd files in order, so it finds them all at once
    for (phase = 1; ok && phase < sc.phases.size(); ++phase) {
        const auto& reqs = sc.phases[phase].reqs;
        for (const auto& q : reqs) send_req(q);
        const unsigned first = next_cmd;
        for (size_t k = reqs.size(); k-- > 0;) {
            next_cmd = first + unsigned(k);
            admit(reqs[k]);
        }
        next_cmd = first + unsigned(reqs.size());
        ok = settle(phase);
    }
    if (!ok && !failed()) error = sc.name + ": timed out";
    atomic_file(dir + "/quit", "");
    t0.join(); t1.join();
    for (int r = 0; r < 2; ++r)
        if (err[r]) try { std::rethrow_exception(err[r]); } catch (const std::exception& e) { error = sc.name + " rank " + i(r) + ": " + e.what(); }
    for (auto& x : s) x.reset();
    if (sc.ends)
        for (const auto& e : by_id) {
            std::vector<std::string> ls;
            std::istringstream in(slurp(dir + "/resp." + u(e.first)));
            for (std::string l; std::getline(in, l);) ls.push_back(l);
            const size_t n = ls.size();
            f0.lines.push_back("resp " + u(e.first) + " " + (n > 1 ? ls[n - 2] : "") + " | " + (n ? ls[n - 1] : ""));
        }
    for (const auto& x : b) remove_dir(x.directory);
    return {f0.lines, f1.lines};
}

std::vector<int> tokens(Rng& r, size_t n) {
    std::vector<int> v(n);
    for (auto& x : v) x = int(r.U(100, 147));
    return v;
}
Req req(uint64_t id, std::vector<int> prompt, int n_gen, const std::string& extra = "") {
    Req q; q.id = id; q.prompt = std::move(prompt);
    q.head = "n_gen " + i(n_gen) + " temp 0 seed " + u(id) + " min 0 stop -" + extra;
    return q;
}

std::vector<Scenario> scenarios() {
    std::vector<Scenario> v;
    Rng r(20261106);
    const bool ds41 = g6mod::has(G6_CAP_PARK);
    {   // a burst of four, a cancel and a resize mid-run, then follow-ups that continue retained runs
        Scenario s; s.name = "reuse";
        s.cfg.pool_rows = 256 * PAGE; s.cfg.max_rows = 16; s.cfg.max_sessions = 4; s.cfg.mtp = 1; s.cfg.prefill_rows = 64;
        s.cfg.two_lanes = false;
        s.reuse = true; s.drained = true;
        if (ds41) { s.seq = true; s.prefill_min = 2; }
        Stage a;
        a.reqs = {req(1, tokens(r, 150), 9), req(2, tokens(r, 90), 30), req(3, tokens(r, 60), 12, " lp 2"), req(4, tokens(r, 20), 5, " plp 1")};
        a.triggers = {Trigger{6, {}, {2}, false}, Trigger{9, {}, {}, true}};
        Stage b;
        std::vector<int> p5 = a.reqs[0].prompt;
        for (int k = 0; k < 60; ++k) p5.push_back(int(r.U(100, 147)));
        b.reqs = {req(5, p5, 7), req(6, tokens(r, 70), 4)};
        s.phases = {a, b};
        v.push_back(s);
    }
    if (!ds41) {   // runs past R on a window pool: a request that does not fit makes a decoding session take the smaller R
        Scenario s; s.name = "window";
        s.cfg.pool_rows = 400 * PAGE; s.cfg.max_rows = 16; s.cfg.max_sessions = 4; s.cfg.mtp = 1; s.cfg.prefill_rows = 128;
        s.cfg.two_lanes = true; s.cfg.window_pages = 4; s.cfg.r2_rows = 4 * PAGE;
        s.pages = 400; s.place = 24; s.below = 8 * PAGE; s.window = true; s.kstop_log = true;
        Stage a;
        a.reqs = {req(11, tokens(r, 700), 40), req(12, tokens(r, 650), 12)};
        a.late = {req(13, tokens(r, 680), 10)};
        a.triggers = {Trigger{8, {13}, {}, false}};
        s.phases = {a};
        v.push_back(s);
    }
    if (!ds41) {   // DFlash2 for a session alone, then a second request that moves it to MTP
        Scenario s; s.name = "dflash";
        s.cfg.pool_rows = 256 * PAGE; s.cfg.max_rows = 24; s.cfg.max_sessions = 4; s.cfg.mtp = 2; s.cfg.prefill_rows = 64;
        s.cfg.guard_tokens = 14; s.cfg.two_lanes = false;
        s.dfl = true;
        Stage a;
        a.reqs = {req(21, tokens(r, 120), 40)};
        a.late = {req(22, tokens(r, 30), 8)};
        a.triggers = {Trigger{12, {22}, {}, false}};
        s.phases = {a};
        v.push_back(s);
    }
    if (ds41) {   // DS41: MTP 0, prefill packs of short prompts, and parked chats that follow-ups resume
        Scenario s; s.name = "park";
        s.cfg.pool_rows = 512 * PAGE; s.cfg.max_rows = 8; s.cfg.max_sessions = 8; s.cfg.mtp = 0; s.cfg.prefill_rows = 64;
        s.cfg.two_lanes = false; s.cfg.c8_pack = true; s.cfg.park = true;
        s.reuse = true; s.park = true; s.seq = true; s.prefill_min = 2;
        Stage a;
        a.reqs = {req(31, tokens(r, 30), 6), req(32, tokens(r, 40), 5), req(33, tokens(r, 25), 7), req(34, tokens(r, 120), 9)};
        a.triggers = {Trigger{5, {}, {34}, false}};
        Stage b;
        std::vector<int> p35 = a.reqs[0].prompt;
        for (int k = 0; k < 20; ++k) p35.push_back(int(r.U(100, 147)));
        b.reqs = {req(35, p35, 6), req(36, tokens(r, 33), 4)};
        s.phases = {a, b};
        v.push_back(s);
    }
    {   // a tight pool: a follow-up continues a retained run and decodes; a long prompt waits for room while it is live,
        // and fits on the pages its cancel gives back
        Scenario s; s.name = "tight";
        s.cfg.pool_rows = 48 * PAGE; s.cfg.max_rows = 16; s.cfg.max_sessions = 4; s.cfg.mtp = 1; s.cfg.prefill_rows = 256;
        s.cfg.two_lanes = false;
        s.pages = 48; s.place = 48; s.below = 48 * 64;
        s.reuse = true;
        if (ds41) { s.seq = true; s.prefill_min = 2; }
        Stage a;
        a.reqs = {req(41, tokens(r, 150), 5)};
        Stage b;
        std::vector<int> p42 = a.reqs[0].prompt;
        for (int k = 0; k < 60; ++k) p42.push_back(int(r.U(100, 147)));
        b.reqs = {req(42, p42, 600)};
        b.late = {req(43, tokens(r, 2900), 6)};
        b.triggers = {Trigger{30, {43}, {}, false}, Trigger{34, {}, {42}, false}};
        s.phases = {a, b};
        v.push_back(s);
    }
    {   // a cancel that lands while its session is still prefilling, beside a session that decodes on
        Scenario s; s.name = "cancel_prefill";
        s.cfg.pool_rows = 256 * PAGE; s.cfg.max_rows = 16; s.cfg.max_sessions = 4; s.cfg.mtp = 1; s.cfg.prefill_rows = 64;
        s.cfg.two_lanes = false;
        s.ends = true;
        if (ds41) { s.seq = true; s.prefill_min = 2; }
        Stage a;
        a.reqs = {req(51, tokens(r, 20), 200), req(52, tokens(r, 1500), 6)};
        a.triggers = {Trigger{8, {}, {52}, false}};
        s.phases = {a};
        v.push_back(s);
    }
    return v;
}

struct Expect { const char* name; uint64_t h[2]; };
// recorded from the service before PoolOps (git 51569016), the pool's plain calls logged; "tight" from the C++ control
// thread before R5b2b (git d07cc4c4); "cancel_prefill" from the C++ compute thread before R5b2c (git b35a5055)
#if G6_MODEL_DS41
const Expect EXPECT[] = {{"reuse", {12273334368408301276ull, 1903311641579599510ull}},
                         {"park", {5353181497655196732ull, 9462988945752174063ull}},
                         {"tight", {2545044441529557428ull, 3958922421392140002ull}},
                         {"cancel_prefill", {10265519460376455732ull, 9550749956254985946ull}}};
#else
const Expect EXPECT[] = {{"reuse", {10598747977024483389ull, 69881064300217143ull}},
                         {"window", {7913264433038465795ull, 13857834022035593827ull}},
                         {"dflash", {3112206073251698158ull, 11351924592827138150ull}},
                         {"tight", {5452963098483769817ull, 10140779659768827361ull}},
                         {"cancel_prefill", {2544064351436808419ull, 13390850078757270789ull}}};
#endif
} // namespace

int main(int argc, char** argv) {
    const bool dump = argc > 1 && !std::strcmp(argv[1], "--dump");
    ::setenv("GR8_S2_ADMIT_HOLD", "1", 1);
    ::setenv("GR8_S2_SHARE_MS", "1", 1);
    // the knob is read once a process: every scenario's head traces to the same file
    const std::string traces = fresh_dir();
    ::setenv("GR8_S2_TRACE", (traces + "/t").c_str(), 1);
    int fails = 0, n = 0;
    const auto all = scenarios();
    require(all.size() == sizeof EXPECT / sizeof EXPECT[0], "a scenario without its hashes");
    for (size_t k = 0; k < all.size(); ++k)
        for (const bool table : {false, true}) {
            const Scenario& sc = all[k];
            std::string error;
            const auto t = run(sc, table, error);
            if (!error.empty()) { std::printf("FAIL %s\n", error.c_str()); ++fails; continue; }
            for (int r = 0; r < 2; ++r) {
                uint64_t h = 14695981039346656037ull;
                for (const auto& l : t[r]) { h = fnv(l.data(), l.size(), h); h = fnv("\n", 1, h); }
                if (dump) for (const auto& l : t[r]) std::printf("%s r%d %s\n", sc.name.c_str(), r, l.c_str());
                const bool ok = sc.name == EXPECT[k].name && h == EXPECT[k].h[r];
                if (!ok) {
                    std::printf("FAIL %s%s rank %d: %zu calls, hash %llu\n", sc.name.c_str(), table ? " (table)" : "", r, t[r].size(),
                                (unsigned long long)h);
                    ++fails;
                }
                ++n;
            }
        }
    remove_dir(traces);
    std::printf(fails ? "module_calls_test: %d FAIL\n" : "module_calls_test: PASS (%d transcripts)\n", fails ? fails : n);
    return fails != 0;
}
