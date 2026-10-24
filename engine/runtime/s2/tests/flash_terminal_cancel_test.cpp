// G6_FLASH_CACHE=terminal and a cancel at the boundary the prefill checkpoint waits for (#174 cd20a51 follow-up 1,
// Assayer reports/assayer-pr174/cd20a51/report.md). The cancel commits at that boundary: the terminal retain keeps
// rows [0, p) at the event (s2/service.h retire_run, flash_native.cuh h.pool.retain), then the native retire takes the
// pending aligned capture (flash_native.cuh h.retire). For a cancel right at the grid point p is the grid point and
// the two have the same tokens, so the second is deduplicated (flash_checkpoint.cuh capture: index_.same) and the
// entry keeps the FIRST capture's flag. Terminal first: the entry is terminal; a later prompt that starts with it
// restores it as not cold-exact (flash_native.cuh bind: exact = !x->terminal) and keeps no aligned checkpoint.
// Before cd20a51 the aligned capture ran inside the quantum, first, and the entry was aligned.
// Asserted here, the pre-cd20a51 outcome: the entry is aligned (terminal=false) and the next prompt keeps its aligned
// checkpoint at its own grid point. Real Service, Driver, scheduler, event gate and reuse state (Dispatch rows) on two
// ranks over socket pairs; the store is the real FlashIndex with capture()'s dedup rule, the ring the real FlashRing;
// the hooks follow flash_native.cuh at cd20a51 (retain, bind, prefill, retire, before_work). No GPU.
// -DBEFORE_CD20A51: the control, the capture inside the quantum as before cd20a51 (the test passes).
#include "s2/service.h"
#include "s2/flash_checkpoint.h"
#include "s2/flash_ring.h"
#include <cstdio>
#include <sys/socket.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

constexpr uint32_t GRID = 2048, P1 = 2 * GRID + 4, P2 = 4 * GRID + 4;   // request 1 checkpoints at 4096, request 2 at 8192

struct Rank {
    std::mutex mu;
    FlashIndex index{100000};
    FlashRing ring;
    uint32_t capture_at = 0;
    std::vector<int> capture_tokens;
    std::atomic<bool> in_q0{false};
    std::atomic<uint32_t> pf_end{0};       // request 2: the end of its last prefill quantum
    std::atomic<long> steps{0};
    std::atomic<int> binds{0};
    bool exact2 = true; uint32_t reuse2 = 0;
    std::vector<std::string> log;
    // flash_checkpoint.cuh FlashCheckpoints::capture: the same tokens already kept -> touch, nothing written
    void capture(uint32_t position, const std::vector<int>& tokens, bool terminal) {
        std::lock_guard<std::mutex> l(mu);
        if (const FlashEntry* x = index.same(tokens, {})) {
            index.touch(x->generation);
            log.push_back((terminal ? "terminal " : "aligned ") + std::to_string(position) + " dedup -> entry " +
                          std::to_string(x->generation) + (x->terminal ? " (terminal)" : " (aligned)"));
            return;
        }
        uint64_t base = 0;
        require(index.reserve(1 + (position + 63) / 64, base), "stand-in store full");
        FlashEntry x;
        x.generation = index.next_generation(); x.base = base; x.blocks = 1 + (position + 63) / 64;
        x.position = position; x.terminal = terminal; x.tokens = tokens;
        log.push_back((terminal ? "terminal " : "aligned ") + std::to_string(position) + " new entry " + std::to_string(x.generation));
        index.add(std::move(x));
    }
    void take_capture() {
        if (capture_tokens.empty()) return;
        capture(uint32_t(capture_tokens.size() - 1), capture_tokens, false);
        capture_tokens.clear();
    }
    NativeHooks hooks() {
        NativeHooks h;
        h.pool.free_pages = [] { return uint32_t(1024); };
        h.pool.room = [] { return uint32_t(1024); };
        h.pool.reserve = [](uint64_t, uint64_t worst) { return Run{0, uint32_t((worst + PAGE - 1) / PAGE), 1, 7}; };
        h.pool.release = [](const Run&) {};
        h.pool.fits = [](uint64_t, uint64_t) { return true; };
        // flash_native.cuh h.pool.retain (terminal), h.pool.match, h.pool.take
        h.pool.retain = [this](const Run&, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace&) {
            if (st.history.size() > 1) {
                const uint32_t p = uint32_t(std::min<int>(st.main_end, int(st.history.size()) - 1));
                int slot = -1;
                try { slot = ring.source(int(p)); } catch (const std::exception&) {}
                if (p > 0 && slot >= 0) capture(p, std::vector<int>(st.history.begin(), st.history.begin() + p + 1), true);
            }
        };
        h.pool.match = [this](const std::vector<int>& prompt, uint64_t, uint64_t, int limit, const g6rt::CacheNamespace& ns) {
            std::lock_guard<std::mutex> l(mu);
            const FlashHit f = index.match(prompt, limit, ns);
            PoolHit hit; hit.lcp = f.lcp; hit.why = f.why;
            if (f.hit) { hit.skip = int(f.position); hit.age = f.generation; }
            return hit;
        };
        h.pool.take = [](const PoolHit&, uint64_t, uint64_t worst) { return Run{0, uint32_t((worst + PAGE - 1) / PAGE), 1, 7}; };
        // flash_native.cuh h.bind (the checkpoint part)
        h.bind = [this](const Grant& g) {
            ring.reset(); capture_at = 0; capture_tokens.clear();
            bool exact = true;
            if (g.reuse) {
                std::lock_guard<std::mutex> l(mu);
                const FlashEntry* x = index.find(match_gen(g));
                require(x && x->position == g.reuse, "hit differs from its entry");
                exact = !x->terminal;
            }
            const uint32_t at = aligned_point(g.req.prompt_tokens, GRID);
            if (exact && at > g.reuse) capture_at = at;
            if (g.req.prompt_tokens == P2) { exact2 = exact; reuse2 = g.reuse; }
            ++binds;
        };
        h.retire = [this](const Slot&) { take_capture(); };
        h.before_work = [this] { take_capture(); };
        h.fence = [] {};
        h.dispatch.bucket = [](bool, int) { return 0; };
        h.dispatch.mode = [](int, int) {};
        h.dispatch.launch = [this](const Launch& l) {
            ++steps;   // the ring is not recorded: only request 2's end-of-request retain reads it after a step
            return std::vector<int>(size_t(l.step.M()), 7);
        };
        h.dispatch.prefill = [this](const Slot& slot, const Payload& p, uint32_t first, uint32_t rows) {
            if (first == 0 && slot.req.prompt_tokens == P1) {   // request 1's quantum 0: long enough to cancel in
                in_q0.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(400));
                in_q0.store(false);
            }
            ++steps;
            if (capture_at && first + rows == capture_at) {
                capture_tokens.assign(p.prompt.begin(), p.prompt.begin() + capture_at + 1);
                capture_at = 0;
#ifdef BEFORE_CD20A51
                take_capture();   // the control: inside the quantum, as cbdcf54 and before
#endif
            }
            if (slot.req.prompt_tokens == P2) pf_end.store(first + rows);
        };
        return h;
    }
    // the generation the head's take() claimed: the newest entry whose tokens start request 2 (match picks it)
    uint64_t match_gen(const Grant& g) {
        uint64_t best = 0;
        for (const auto& x : index.entries()) if (x.position == g.reuse && x.generation > best) best = x.generation;
        return best;
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
    CHECK(wait([&] { std::lock_guard<std::mutex> a(r[0].mu), b(r[1].mu); return r[0].index.size() && r[1].index.size(); }));
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
        for (const auto& l : x.log) std::printf("rank %d: %s\n", i, l.c_str());
        const FlashEntry* e1 = nullptr; const FlashEntry* e2 = nullptr;
        for (const auto& e : x.index.entries()) { if (e.position == 2 * GRID) e1 = &e; if (e.position == 4 * GRID) e2 = &e; }
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
