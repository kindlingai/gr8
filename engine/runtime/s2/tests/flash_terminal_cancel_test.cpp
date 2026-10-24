// G6_FLASH_CACHE=terminal and a cancel at the boundary the prefill checkpoint waits for (#174 cd20a51 follow-up 1,
// Assayer reports/assayer-pr174/cd20a51/report.md). The cancel commits at that boundary: the terminal retain keeps
// rows [0, p) at the event (s2/service.h retire_run, flash_native.cuh h.pool.retain) and the native retire takes the
// pending aligned capture (flash_native.cuh h.retire). For a cancel right at the grid point the two have the same
// tokens, so the second is deduplicated (flash_checkpoint.cuh capture: index_.same) and the entry keeps the FIRST
// capture's flag. Terminal first: the entry is terminal, and a later prompt that starts with it restores it as not
// cold-exact (flash_native.cuh bind: exact = !x->terminal) and keeps no aligned checkpoint (cd20a51). Asserted: the
// entry is aligned and the next prompt keeps its aligned checkpoint at its own grid point (before cd20a51, and with
// the retain flushing the pending capture first).
// Real Service, Driver, scheduler, event gate and reuse state (Dispatch rows) on two ranks over socket pairs. The
// Flash hooks are flash_native.cuh's own source -- take_capture, h.pool.retain, the prefill's capture block, h.retire,
// h.before_work -- pasted in at the @...@ markers by engine/tests/test_flash_terminal_cancel.py, over a store that is
// the real FlashIndex with capture()'s dedup rule and the real FlashRing. bind is this file's copy of the checkpoint
// part of flash_native.cuh h.bind (the rest is GPU). No GPU. Run: python3 engine/tests/test_flash_terminal_cancel.py
#include "s2/service.h"
#include "s2/flash_checkpoint.h"
#include "s2/flash_ring.h"
#include <cstdio>
#include <sys/socket.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

constexpr uint32_t GRID = 2048, P1 = 2 * GRID + 4, P2 = 4 * GRID + 4;   // request 1 checkpoints at 4096, request 2 at 8192

// the FlashCheckpoints calls the hooks make, over the real index (flash_checkpoint.cuh capture: dedup, then a new entry)
struct Ckpt {
    FlashIndex ix{100000};
    struct Last { uint64_t blocks = 0; double ms = 0; } last;
    std::vector<std::string> log;
    FlashIndex& index() { return ix; }
    uint64_t blocks() const { return ix.blocks(); }
    uint64_t capture(uint32_t position, int, const std::vector<int>& tokens, const g6rt::CacheNamespace& ns, int,
                     bool terminal = false) {
        if (const FlashEntry* x = ix.same(tokens, ns)) {
            ix.touch(x->generation);
            log.push_back((terminal ? "terminal " : "aligned ") + std::to_string(position) + " dedup -> entry " +
                          std::to_string(x->generation) + (x->terminal ? " (terminal)" : " (aligned)"));
            return x->generation;
        }
        uint64_t base = 0;
        const uint64_t need = 1 + (position + 63) / 64;
        if (!ix.reserve(need, base)) return 0;
        FlashEntry x;
        x.generation = ix.next_generation(); x.base = base; x.blocks = need; x.position = position; x.terminal = terminal;
        x.tokens = tokens; x.ns = ns;
        log.push_back((terminal ? "terminal " : "aligned ") + std::to_string(position) + " new entry " + std::to_string(x.generation));
        const uint64_t g = x.generation;
        ix.add(std::move(x));
        return g;
    }
};
struct Rank {
    struct { bool terminal = true; int stream = 0; } c;   // G6_FLASH_CACHE=terminal
    Ckpt real; Ckpt* ck = &real;
    struct { void release(const Run&) {} } pool;
    FlashRing ring;
    std::mutex ck_mu;
    uint32_t capture_at = 0;
    g6rt::CacheNamespace capture_ns{};
    std::vector<int> capture_tokens;
    uint64_t capture_req = 0;
    std::function<void()> take_capture;
    void fence() {}
    std::atomic<bool> in_q0{false};
    std::atomic<uint32_t> pf_end{0};       // request 2: the end of its last prefill quantum
    std::atomic<long> steps{0};
    bool exact2 = true; uint32_t reuse2 = 0;
    NativeHooks hooks() {
        NativeHooks h;
        // flash_native.cuh take_capture
@TAKE_CAPTURE@
        h.pool.free_pages = [] { return uint32_t(1024); };
        h.pool.room = [] { return uint32_t(1024); };
        h.pool.reserve = [](uint64_t, uint64_t worst) { return Run{0, uint32_t((worst + PAGE - 1) / PAGE), 1, 7}; };
        h.pool.release = [](const Run&) {};
        h.pool.fits = [](uint64_t, uint64_t) { return true; };
        // flash_native.cuh h.pool.retain
@RETAIN@
        h.pool.match = [this](const std::vector<int>& prompt, uint64_t, uint64_t, int limit, const g6rt::CacheNamespace& ns) {
            std::lock_guard<std::mutex> l(ck_mu);
            const FlashHit f = ck->index().match(prompt, limit, ns);
            PoolHit hit; hit.lcp = f.lcp; hit.why = f.why;
            if (f.hit) { hit.skip = int(f.position); hit.age = f.generation; }
            return hit;
        };
        h.pool.take = [](const PoolHit&, uint64_t, uint64_t worst) { return Run{0, uint32_t((worst + PAGE - 1) / PAGE), 1, 7}; };
        // the checkpoint part of flash_native.cuh h.bind
        h.bind = [this](const Grant& g) {
            ring.reset(); capture_at = 0; capture_tokens.clear();
            capture_ns = g6rt::steering::cache_namespace(g.req.steering);
            bool exact = true;
            if (g.reuse) {
                std::lock_guard<std::mutex> l(ck_mu);
                const FlashEntry* x = nullptr;   // the newest entry at the hit's position (the one match picks)
                for (const auto& e : ck->index().entries()) if (e.position == g.reuse && (!x || e.generation > x->generation)) x = &e;
                require(x && x->position == g.reuse, "hit differs from its entry");
                exact = !x->terminal;
            }
            const uint32_t at = aligned_point(g.req.prompt_tokens, GRID);
            if (exact && at > g.reuse) capture_at = at;
            if (g.req.prompt_tokens == P2) { exact2 = exact; reuse2 = g.reuse; }
        };
        // flash_native.cuh h.retire, h.before_work
@RETIRE@
@BEFORE_WORK@
        h.dispatch.bucket = [](bool, int) { return 0; };
        h.dispatch.mode = [](int, int) {};
        h.dispatch.launch = [this](const Launch& l) {
            ++steps;   // the ring is not recorded: only request 2's end-of-request retain reads it after a step
            return std::vector<int>(size_t(l.step.M()), 7);
        };
        h.fence = [] {};
        h.dispatch.prefill = [this](const Slot& slot, const Payload& p, uint32_t first, uint32_t rows) {
            if (first == 0 && slot.req.prompt_tokens == P1) {   // request 1's quantum 0: long enough to cancel in
                in_q0.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(400));
                in_q0.store(false);
            }
            ++steps;
            // flash_native.cuh h.dispatch.prefill: the capture block
@PREFILL_CAPTURE@
            if (slot.req.prompt_tokens == P2) pf_end.store(first + rows);
        };
        return h;
    }
};

static std::string dir() {
    char t[] = "/tmp/flash_terminal_cancel_XXXXXX";
    require(::mkdtemp(t) != nullptr, "mkdtemp");
    return t;
}
static std::string ids(uint32_t n) {
    std::string s;
    for (uint32_t i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(100 + i % 1000);
    return s;
}

int main() {
    ::setenv("G6_S2_EVENTLOG", "1", 1);
    Config cfg;
    cfg.pool_rows = 64 * 1024; cfg.max_rows = 3; cfg.max_sessions = 1; cfg.mtp = 2;
    cfg.prefill_rows = GRID; cfg.two_lanes = false;
    int sp[2];
    require(!::socketpair(AF_UNIX, SOCK_STREAM, 0, sp), "socketpair");
    ControlLinks head_links, worker_links;
    head_links.links.push_back(std::make_unique<ControlIO>(sp[0]));
    worker_links.links.push_back(std::make_unique<ControlIO>(sp[1]));
    for (auto* l : {&head_links, &worker_links}) l->links[0]->timeout_seconds(20);
    BootstrapConfig b0, b1;
    b0.scheduler = b1.scheduler = cfg;
    b0.ranks = b1.ranks = 2; b0.rank = 0; b1.rank = 1;
    b0.head_ip = b1.head_ip = "127.0.0.1"; b0.port = b1.port = 1;
    b0.runtime_signature = b1.runtime_signature = "flash_terminal_cancel_test";
    b0.directory = dir(); b1.directory = dir();
    Rank r[2];
    std::exception_ptr err[2];
    Service s0(b0, head_links, r[0].hooks(), 1), s1(b1, worker_links, r[1].hooks(), 1);
    s0.quit_linger_ms = s1.quit_linger_ms = 300;
    std::thread t0([&] { try { s0.run(); } catch (...) { err[0] = std::current_exception(); } });
    std::thread t1([&] { try { s1.run(); } catch (...) { err[1] = std::current_exception(); } });
    auto wait = [&](const std::function<bool()>& f) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (!f() && std::chrono::steady_clock::now() < end && !err[0] && !err[1]) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return f();
    };
    // request 1 (P1 tokens), cancelled while its quantum 0 runs: the cancel lands at the boundary quantum 1 ends on
    const std::string q1 = "n_gen 8 temp 0 seed 1 min 0 stop -\n" + ids(P1) + "\n";
    atomic_file(b0.directory + "/req.1", q1);
    atomic_file(b0.directory + "/cmd.0", "ADMIT 1 1 " + std::to_string(payload_hash(q1)) + "\n");
    CHECK(wait([&] { return r[0].in_q0.load(); }));
    atomic_file(b0.directory + "/cmd.1", "CANCEL 1 1 0\n");
    CHECK(wait([&] { std::lock_guard<std::mutex> a(r[0].ck_mu), b(r[1].ck_mu); return r[0].real.ix.size() && r[1].real.ix.size(); }));
    // request 2 (P2 tokens) starts with request 1's first 4097 tokens: it restores the 4096 entry and should keep 8192
    const std::string q2 = "n_gen 2 temp 0 seed 1 min 0 stop -\n" + ids(P2) + "\n";
    atomic_file(b0.directory + "/req.2", q2);
    atomic_file(b0.directory + "/cmd.2", "ADMIT 1 2 " + std::to_string(payload_hash(q2)) + "\n");
    CHECK(wait([&] { return r[0].pf_end.load() >= 4 * GRID && r[1].pf_end.load() >= 4 * GRID &&
                            r[0].steps.load() >= 6 && r[1].steps.load() >= 6; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));   // the boundary after the 8192 quantum
    atomic_file(b0.directory + "/quit", "");
    t0.join(); t1.join();
    for (int i = 0; i < 2; ++i)
        if (err[i]) try { std::rethrow_exception(err[i]); } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %d: %s\n", i, e.what()); ++fails;
        }
    for (int i = 0; i < 2; ++i) {
        auto& x = r[i];
        for (const auto& l : x.real.log) std::printf("rank %d: %s\n", i, l.c_str());
        const FlashEntry* e1 = nullptr; const FlashEntry* e2 = nullptr;
        for (const auto& e : x.real.ix.entries()) { if (e.position == 2 * GRID) e1 = &e; if (e.position == 4 * GRID) e2 = &e; }
        std::printf("rank %d: entry at %u %s; request 2 restored %u rows, %s, %s an aligned entry at %u\n", i, 2 * GRID,
                    e1 ? (e1->terminal ? "terminal" : "aligned") : "missing", x.reuse2, x.exact2 ? "exact" : "NOT exact",
                    e2 ? "kept" : "did NOT keep", 4 * GRID);
        CHECK(e1 && !e1->terminal);   // the cancelled prompt's grid-point entry is aligned, as before cd20a51
        CHECK(x.reuse2 == 2 * GRID && x.exact2);
        CHECK(e2 && !e2->terminal);   // request 2 keeps its own aligned checkpoint
    }
    std::printf(fails ? "flash_terminal_cancel_test: %d FAIL\n" : "flash_terminal_cancel_test: PASS\n", fails);
    return fails != 0;
}
