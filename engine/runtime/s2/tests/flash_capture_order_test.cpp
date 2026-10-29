// The Flash prefill checkpoint and a Resize on its tier (D-331 M8 box3 p4 k0): the head announces a grow of the
// checkpoint store while quantum 0 of a prompt is running; the notice lands NOTICE_MIN boundaries on, which is the
// boundary quantum 1 ends on -- the prompt's last grid point, where the checkpoint is taken. The capture must see the
// grown store: a full store of 5 entries keeps 6 after the capture, not 5 (one dropped for room).
// The real Service (service.h: controller, event lookahead, compute loop, commit order), Driver and scheduler on two
// ranks over socket pairs; the native layer is a stand-in with no GPU: a store that counts entries and evicts one when
// full, and the capture placed where flash_native.cuh places it on the tree under test -- in NativeHooks::before_work
// when the tree has it (after the boundary's commits), else at the end of the prefill quantum (cbdcf54 and before).
// Build and run: see scripts/host_tests.sh.
#include "s2/service.h"
#include <cstdio>
#include <sys/socket.h>
#include <sys/stat.h>
#include <type_traits>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

template<class H, class = void> struct has_before_work : std::false_type {};
template<class H> struct has_before_work<H, std::void_t<decltype(std::declval<H&>().before_work)>> : std::true_type {};
template<class H, class F> static void set_before_work(H& h, F f) {
    if constexpr (has_before_work<H>::value) h.before_work = f; else (void)h, (void)f;
}

constexpr uint32_t GRID = 2048, PROMPT = 2 * GRID + 4, CAPTURE = 2 * GRID;   // capture at the last grid point
constexpr uint32_t ENTRIES = 5;                                                  // the store is full at 5 before the grow

// one rank's native layer
struct Rank {
    std::mutex mu;
    uint32_t capacity = ENTRIES, entries = ENTRIES, dropped = 0;
    uint64_t boundary = 0;                     // boundaries departed (prefill quanta and steps counted at the hook)
    long resize_at = -1, capture_at = -1;      // order: the step count when each happened
    uint32_t kept = 0, seen = 0;               // entries right after the capture; the store size it saw
    std::atomic<long> step{0};
    bool due = false;
    std::atomic<bool> in_q0{false}, q0_done{false};
    void capture() {
        std::lock_guard<std::mutex> l(mu);
        if (entries == capacity) { --entries; ++dropped; }
        seen = capacity; ++entries; kept = entries; capture_at = step;
    }
    NativeHooks hooks(bool head) {
        NativeHooks h;
        h.pool.free_pages = [] { return uint32_t(1024); };
        h.pool.room = [] { return uint32_t(1024); };
        h.pool.reserve = [](uint64_t, uint64_t worst) { return Run{0, uint32_t((worst + PAGE - 1) / PAGE), 1, 7}; };
        h.pool.release = [](const Run&) {};
        h.pool.fits = [](uint64_t, uint64_t) { return true; };
        h.bind = [this](const Grant&) { due = false; };
        h.retire = [](const Slot&) {};
        h.fence = [] {};
        h.dispatch.bucket = [](bool, int) { return 0; };
        h.dispatch.mode = [](int, int) {};
        h.dispatch.launch = [this](const Launch& l) { ++step; return std::vector<int>(size_t(l.step.M()), 7); };
        h.dispatch.prefill = [this](const Slot&, const Payload&, uint32_t first, uint32_t rows) {
            if (first == 0) {   // quantum 0 runs long enough for the head's share poll to announce the grow
                in_q0.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(400));
                in_q0.store(false); q0_done.store(true);
            }
            ++step;
            if (first + rows == CAPTURE) {
                if constexpr (has_before_work<NativeHooks>::value) due = true;
                else capture();   // cbdcf54: inside the quantum, before the boundary's commits
            }
        };
        set_before_work(h, [this] { if (due) { due = false; capture(); } });
        h.resize = [this](const Resize& r) {
            std::lock_guard<std::mutex> l(mu);
            capacity = uint32_t(r.slots); resize_at = step;
        };
        if (head) {
            auto asked = std::make_shared<bool>(false);
            h.share = [this, asked](bool idle) -> std::optional<Resize> {
                if (idle || *asked || !in_q0.load()) return std::nullopt;
                *asked = true;
                return Resize{1, ENTRIES + 1, ENTRIES + 1, 1};   // grow by one entry
            };
        }
        return h;
    }
};

static std::string dir() {
    char t[] = "/tmp/flash_capture_order_XXXXXX";
    require(::mkdtemp(t) != nullptr, "mkdtemp");
    return t;
}

int main() {
    ::setenv("GR8_S2_SHARE_MS", "10", 1);
    ::setenv("GR8_S2_EVENTLOG", "1", 1);
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
    b0.runtime_signature = b1.runtime_signature = "flash_capture_order_test";
    b0.directory = dir(); b1.directory = dir();
    Rank r[2];
    std::exception_ptr err[2];
    Service s0(b0, head_links, r[0].hooks(true), 1), s1(b1, worker_links, r[1].hooks(false), 1);
    s0.quit_linger_ms = s1.quit_linger_ms = 300;
    std::thread t0([&] { try { s0.run(); } catch (...) { err[0] = std::current_exception(); } });
    std::thread t1([&] { try { s1.run(); } catch (...) { err[1] = std::current_exception(); } });

    // one request: PROMPT tokens, 2 generated
    std::string prompt;
    for (uint32_t i = 0; i < PROMPT; ++i) prompt += (i ? "," : "") + std::to_string(100 + i % 1000);
    const std::string bytes = "n_gen 2 temp 0 seed 1 min 0 stop -\n" + prompt + "\n";
    atomic_file(b0.directory + "/req.1", bytes);
    atomic_file(b0.directory + "/cmd.0", "ADMIT 1 1 " + std::to_string(payload_hash(bytes)) + "\n");
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    // done: both ranks past the capture and into the prompt's tail (the steps after the last grid point)
    auto past = [&](Rank& x) { std::lock_guard<std::mutex> l(x.mu); return x.capture_at >= 0 && x.step.load() >= 4; };
    while ((!past(r[0]) || !past(r[1])) && std::chrono::steady_clock::now() < end && !err[0] && !err[1])
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    atomic_file(b0.directory + "/quit", "");
    t0.join(); t1.join();
    for (int i = 0; i < 2; ++i)
        if (err[i]) try { std::rethrow_exception(err[i]); } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %d: %s\n", i, e.what()); ++fails;
        }

    std::printf("flash_capture_order_test: placement %s\n",
                has_before_work<NativeHooks>::value ? "before_work (after the boundary's commits)" : "in the prefill quantum");
    for (int i = 0; i < 2; ++i) {
        auto& x = r[i];
        std::printf("rank %d: %ld steps, resize at step %ld, capture at step %ld saw a store of %u, kept %u entries (%u dropped)\n",
                    i, x.step.load(), x.resize_at, x.capture_at, x.seen, x.kept, x.dropped);
        CHECK(x.step.load() >= 4);
        CHECK(x.q0_done.load());
        CHECK(x.resize_at >= 0 && x.capture_at >= 0);
        CHECK(x.resize_at == 2);                  // the grow commits at the boundary quantum 1 ends on (2 quanta run)
        CHECK(x.seen == ENTRIES + 1);             // ... before the capture taken there
        CHECK(x.capacity == ENTRIES + 1);
        CHECK(x.kept == ENTRIES + 1);             // p4 k0: 6 entries kept, not 5
        CHECK(x.dropped == 0);
    }
    CHECK(r[0].kept == r[1].kept && r[0].capture_at == r[1].capture_at);
    std::printf(fails ? "flash_capture_order_test: %d FAIL\n" : "flash_capture_order_test: PASS\n", fails);
    return fails != 0;
}
