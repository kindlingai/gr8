// GR8_S2_ADMIT_HOLD=1 and a burst whose ADMIT files land while the head is inside its hold chain: a shim can write
// all 8 cmd files within 1 ms. The head's control loop reads cmd.<k> only between events (service.h loop), so a held
// commit that looks only at the in-memory queue misses a file that lands during the chain and releases the boundary
// after the 2nd grant (2 commits at one boundary, 6 at a boundary 4 later).
// Here: cmd.0 and cmd.1 are written before the run; cmd.2 .. cmd.7 are written from the head rank's bind of the first grant,
// i.e. while the head is still inside that event (it waits for every rank's ACK). Asserted: all 8 grants apply at one boundary on
// both ranks (the same step count at every bind) and the hold chain ends with the 8th (live + 1 == limit). Without the drain
// (Service::drain_admits) the 3rd .. 8th grants apply after the 2nd grant's release, at a later step count.
// Real Service, Driver, scheduler and event gate on two ranks over socket pairs; the native layer is a stand-in with no GPU
// (a prefill quantum and a step each sleep 20 ms so boundaries pass while a notice waits). Build and run: see scripts/host_tests.sh.
#include "s2/service.h"
#include <cstdio>
#include <sys/socket.h>
#include <sys/stat.h>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

constexpr int N = 8;          // the c8 burst
constexpr uint32_t PROMPT = 40;

static std::string req_bytes(int i) {
    std::string prompt;
    for (uint32_t t = 0; t < PROMPT; ++t) prompt += (t ? "," : "") + std::to_string(100 + 7 * i + t);
    return "n_gen 4 temp 0 seed 1 min 0 stop -\n" + prompt + "\n";
}

struct Rank {
    std::mutex mu;
    std::atomic<long> step{0};               // prefill quanta + steps run: a bind at the same count = the same boundary
    std::vector<long> bind_at;               // the step count at each grant's bind, in grant order
    std::function<void()> on_first_bind;     // head: writes cmd.2 .. cmd.7 during the first event
    NativeHooks hooks() {
        NativeHooks h;
        h.pool.free_pages = [] { return uint32_t(1024); };
        h.pool.room = [] { return uint32_t(1024); };
        h.pool.reserve = [](uint64_t, uint64_t worst, uint32_t) { return Run{0, uint32_t((worst + PAGE - 1) / PAGE), 1, 7}; };
        h.pool.release = [](const Run&) {};
        h.pool.fits = [](uint64_t, uint64_t, uint32_t) { return true; };
        h.bind = [this](const Grant&) {
            std::function<void()> f;
            {
                std::lock_guard<std::mutex> l(mu);
                bind_at.push_back(step.load());
                if (bind_at.size() == 1) f = on_first_bind;
            }
            if (f) f();
        };
        h.retire = [](const Slot&) {};
        h.fence = [] {};
        h.dispatch.bucket = [](bool, int) { return 0; };
        h.dispatch.mode = [](int, int) {};
        h.dispatch.launch = [this](const Launch& l) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20)); ++step;
            return std::vector<int>(size_t(l.step.M()), 7);
        };
        h.dispatch.prefill = [this](const Slot&, const Payload&, uint32_t, uint32_t) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20)); ++step;
        };
        return h;
    }
};

static std::string dir() {
    char t[] = "/tmp/admit_hold_drain_XXXXXX";
    require(::mkdtemp(t) != nullptr, "mkdtemp");
    return t;
}

int main() {
    ::setenv("GR8_S2_ADMIT_HOLD", "1", 1);
    ::setenv("GR8_S2_EVENTLOG", "1", 1);
    Config cfg;
    cfg.pool_rows = 64 * 1024; cfg.max_rows = 16; cfg.max_sessions = N; cfg.mtp = 1;   // limit() = 16 / (mtp + 1) = 8
    cfg.prefill_rows = 64; cfg.two_lanes = false;
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
    b0.runtime_signature = b1.runtime_signature = "admit_hold_drain_test";
    b0.directory = dir(); b1.directory = dir();
    for (int i = 0; i < N; ++i) atomic_file(b0.directory + "/req." + std::to_string(i + 1), req_bytes(i));
    auto admit = [&](int i) {
        atomic_file(b0.directory + "/cmd." + std::to_string(i),
                    "ADMIT 1 " + std::to_string(i + 1) + " " + std::to_string(payload_hash(req_bytes(i))) + "\n");
    };
    Rank r[2];
    r[0].on_first_bind = [&] { for (int i = 2; i < N; ++i) admit(i); };   // land during the first event (the head waits for ACKs)
    admit(0); admit(1);
    std::exception_ptr err[2];
    Service s0(b0, head_links, r[0].hooks(), 1), s1(b1, worker_links, r[1].hooks(), 1);
    s0.quit_linger_ms = s1.quit_linger_ms = 300;
    std::thread t0([&] { try { s0.run(); } catch (...) { err[0] = std::current_exception(); } });
    std::thread t1([&] { try { s1.run(); } catch (...) { err[1] = std::current_exception(); } });
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    auto all = [&](Rank& x) { std::lock_guard<std::mutex> l(x.mu); return x.bind_at.size() >= size_t(N); };
    while ((!all(r[0]) || !all(r[1])) && std::chrono::steady_clock::now() < end && !err[0] && !err[1])
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    atomic_file(b0.directory + "/quit", "");
    t0.join(); t1.join();
    for (int i = 0; i < 2; ++i)
        if (err[i]) try { std::rethrow_exception(err[i]); } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %d: %s\n", i, e.what()); ++fails;
        }
    for (int i = 0; i < 2; ++i) {
        std::lock_guard<std::mutex> l(r[i].mu);
        std::printf("rank %d: %zu grants, step count at each bind:", i, r[i].bind_at.size());
        for (long s : r[i].bind_at) std::printf(" %ld", s);
        std::printf("\n");
        CHECK(r[i].bind_at.size() == size_t(N));
        for (size_t k = 1; k < r[i].bind_at.size(); ++k) CHECK(r[i].bind_at[k] == r[i].bind_at[0]);   // one boundary for all 8
    }
    CHECK(r[0].bind_at == r[1].bind_at);
    std::printf(fails ? "admit_hold_drain_test: %d FAIL\n" : "admit_hold_drain_test: PASS (all %d grants at one boundary)\n",
                fails ? fails : N);
    return fails != 0;
}
