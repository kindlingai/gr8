// End-to-end host tests for EL8 in-process live width switching (#156):
//  1. LiveSwitchCoordinator 5-step happy path with pre-capture (4 -> 3): verifies strict step ordering,
//     zero deferred capture at fence, online defrag armed, zero leaked resources.
//  2. LiveSwitchCoordinator deferred-capture path when VRAM headroom is tight (3 -> 4): verifies precapture is
//     skipped, old graphs are released BEFORE target graphs are captured at the fence, and defrag is armed.
//  3. LiveSwitchCoordinator pre-cutover failure & rollback (unsupported width, failure in stage_weights,
//     open_fabric_group, migrate_kv_bulk, precapture_graphs, and explicit abort): verifies reverse-order rollback
//     of only staged resources, active width untouched, and cutover refused on an aborted plan.
//  4. Full multi-rank Service loopback test:
//     - 2 ranks actively decode 2 concurrent sessions (req 1 and req 2).
//     - Mid-decode, the head writes `SWITCH <epoch> <group_id> <target_nr>` to the S2 ingress directory.
//     - Both ranks hit the synchronized EventKind::Switch fence, execute NativeHooks::switch_width (driving
//       LiveSwitchCoordinator::cutover_at_fence + finish_and_defrag), leave both live sessions resident with
//       zero retires / zero errors / zero prefill replays, and both sessions emit all expected tokens.
// Host only, no CUDA; runs under ASan + UBSan in scripts/host_tests.sh.
#include "../live_switch.h"
#include "../service.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace g6s2;

namespace {
int g_fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fails; } } while (0)

std::string mktemp_dir(const char* prefix) {
    std::string tmpl = std::string("/tmp/") + prefix + "XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    require(::mkdtemp(buf.data()) != nullptr, "mkdtemp");
    return std::string(buf.data());
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}

struct MockSwitchState {
    uint32_t active_nr = 4;
    uint64_t active_group = 1;
    int staged_weights = 0;
    int open_fabric_groups = 1;
    int active_migrations = 0;
    int old_graphs = 1;
    int target_graphs = 0;
    int defrag_armed = 0;
    int fail_at = 0; // 1=weights, 2=fabric, 3=migrate, 4=precapture
    std::vector<std::string> trace;

    LiveSwitchOps ops() {
        LiveSwitchOps o;
        o.has_width = [this](uint32_t nr) {
            trace.push_back("has_width:" + std::to_string(nr));
            return nr == 3 || nr == 4 || nr == 6;
        };
        o.announce_group = [this](const SwitchPlan& p) {
            trace.push_back("announce:" + std::to_string(p.group_id));
        };
        o.stage_weights = [this](const SwitchPlan&) {
            trace.push_back("stage_weights");
            if (fail_at == 1) throw std::runtime_error("injected weight stage error");
            ++staged_weights;
        };
        o.open_fabric_group = [this](const SwitchPlan&) {
            trace.push_back("open_fabric");
            if (fail_at == 2) throw std::runtime_error("injected fabric open error");
            ++open_fabric_groups;
        };
        o.migrate_kv_bulk = [this](const SwitchPlan&) -> uint32_t {
            trace.push_back("migrate_bulk");
            ++active_migrations;
            if (fail_at == 3) throw std::runtime_error("injected migrate bulk error");
            return 4;
        };
        o.precapture_graphs = [this](const SwitchPlan&) {
            trace.push_back("precapture");
            if (fail_at == 4) throw std::runtime_error("injected precapture OOM");
            ++target_graphs;
        };
        o.fence_tail_kv = [this](const SwitchPlan&) {
            trace.push_back("fence_tail_kv");
            active_migrations = 0;
        };
        o.release_old_graphs = [this]() {
            trace.push_back("release_old_graphs");
            old_graphs = 0;
        };
        o.capture_target_graphs = [this](const SwitchPlan&) {
            trace.push_back("capture_target_graphs");
            ++target_graphs;
        };
        o.swap_active_width = [this](const SwitchPlan& p) {
            trace.push_back("swap_width:" + std::to_string(p.target_nr));
            active_nr = p.target_nr;
            active_group = p.group_id;
        };
        o.release_old_weights = [this](const SwitchPlan&) {
            trace.push_back("release_old_weights");
            staged_weights = 0;
        };
        o.close_old_fabric_group = [this](const SwitchPlan&) {
            trace.push_back("close_old_fabric");
            --open_fabric_groups;
        };
        o.arm_online_defrag = [this](const SwitchPlan&) {
            trace.push_back("arm_defrag");
            ++defrag_armed;
        };
        o.rollback_precaptured_graphs = [this](const SwitchPlan&) {
            trace.push_back("rb_precapture");
            --target_graphs;
        };
        o.rollback_kv_migration = [this](const SwitchPlan&) {
            trace.push_back("rb_migrate");
            --active_migrations;
        };
        o.rollback_fabric_group = [this](const SwitchPlan&) {
            trace.push_back("rb_fabric");
            --open_fabric_groups;
        };
        o.rollback_staged_weights = [this](const SwitchPlan&) {
            trace.push_back("rb_weights");
            --staged_weights;
        };
        return o;
    }
};

void test_coordinator_precapture_happy_path() {
    MockSwitchState st;
    LiveSwitchCoordinator coord(st.ops());
    SwitchPlan p{42, 4, 3, 0, /*need_capture_bytes=*/512ull << 20, /*avail_vram_bytes=*/2048ull << 20};
    CHECK(coord.prepare(p));
    CHECK(coord.phase() == SwitchPhase::PreCaptured);
    CHECK(coord.stats().precaptured);
    CHECK(coord.stats().migrated_sessions == 4);
    CHECK(st.active_nr == 4 && st.staged_weights == 1 && st.open_fabric_groups == 2 && st.target_graphs == 1);

    coord.cutover_at_fence(SwitchOp{42, 3, 0});
    CHECK(coord.phase() == SwitchPhase::Cutover);
    CHECK(st.active_nr == 3 && st.active_group == 42 && st.old_graphs == 0 && st.target_graphs == 1);

    coord.finish_and_defrag();
    CHECK(coord.phase() == SwitchPhase::Completed);
    CHECK(coord.stats().defrag_armed);
    CHECK(st.staged_weights == 0 && st.open_fabric_groups == 1 && st.active_migrations == 0 && st.defrag_armed == 1);

    const std::vector<std::string> want = {
        "has_width:3", "announce:42", "stage_weights", "open_fabric", "migrate_bulk", "precapture",
        "fence_tail_kv", "release_old_graphs", "swap_width:3",
        "release_old_weights", "close_old_fabric", "arm_defrag",
    };
    CHECK(st.trace == want);
}

void test_coordinator_deferred_capture_path() {
    MockSwitchState st;
    st.active_nr = 3;
    LiveSwitchCoordinator coord(st.ops());
    SwitchPlan p{99, 3, 4, 0, /*need_capture_bytes=*/2048ull << 20, /*avail_vram_bytes=*/256ull << 20};
    CHECK(!p.can_precapture());
    CHECK(coord.prepare(p));
    CHECK(!coord.stats().precaptured);
    CHECK(st.target_graphs == 0);

    coord.cutover_at_fence(SwitchOp{99, 4, 0});
    CHECK(st.active_nr == 4 && st.old_graphs == 0 && st.target_graphs == 1);
    coord.finish_and_defrag();
    CHECK(coord.phase() == SwitchPhase::Completed && st.open_fabric_groups == 1 && st.defrag_armed == 1);

    const std::vector<std::string> want = {
        "has_width:4", "announce:99", "stage_weights", "open_fabric", "migrate_bulk",
        "fence_tail_kv", "release_old_graphs", "capture_target_graphs", "swap_width:4",
        "release_old_weights", "close_old_fabric", "arm_defrag",
    };
    CHECK(st.trace == want);
}

void test_coordinator_rollbacks() {
    // Unsupported width
    {
        MockSwitchState st;
        LiveSwitchCoordinator coord(st.ops());
        CHECK(!coord.prepare(SwitchPlan{7, 4, 5, 0, 0, 0}));
        CHECK(coord.phase() == SwitchPhase::Aborted && coord.stats().rolled_back);
        CHECK(st.active_nr == 4 && st.staged_weights == 0 && st.open_fabric_groups == 1 && st.target_graphs == 0);
        bool threw = false;
        try { coord.cutover_at_fence(SwitchOp{7, 5, 0}); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }
    // Failure injected at each Step 2/3 substep rolls back only what was staged, in reverse order
    for (int fail_at = 1; fail_at <= 4; ++fail_at) {
        MockSwitchState st;
        st.fail_at = fail_at;
        LiveSwitchCoordinator coord(st.ops());
        CHECK(!coord.prepare(SwitchPlan{55, 4, 3, 0, 100, 1000}));
        CHECK(coord.phase() == SwitchPhase::Aborted && coord.stats().rolled_back);
        CHECK(st.active_nr == 4);
        CHECK(st.staged_weights == 0);
        CHECK(st.open_fabric_groups == 1);
        CHECK(st.active_migrations == 0);
        CHECK(st.target_graphs == 0);
        CHECK(st.old_graphs == 1);
    }
    // Explicit abort after prepare (including pre-captured graphs)
    {
        MockSwitchState st;
        LiveSwitchCoordinator coord(st.ops());
        CHECK(coord.prepare(SwitchPlan{77, 4, 3, 0, 100, 1000}));
        CHECK(st.target_graphs == 1 && st.staged_weights == 1 && st.open_fabric_groups == 2);
        coord.abort("peer failed prepare");
        CHECK(coord.phase() == SwitchPhase::Aborted && coord.stats().rolled_back);
        CHECK(st.active_nr == 4 && st.staged_weights == 0 && st.open_fabric_groups == 1 &&
              st.active_migrations == 0 && st.target_graphs == 0 && st.old_graphs == 1);
    }
}

// ---------- Test 4: Full 2-rank Service loopback live SWITCH mid-decode ----------
void write_req_and_admit(const std::string& dir, uint64_t epoch, uint64_t cmd_idx, uint64_t req_id,
                         const std::vector<int>& prompt, int max_tokens) {
    std::ostringstream oss;
    oss << "n_gen " << max_tokens << " temp 0 seed 1 min 0 stop -\n";
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (i) oss << ',';
        oss << prompt[i];
    }
    oss << "\n";
    const std::string body = oss.str();
    atomic_file(dir + "/req." + std::to_string(req_id), body);
    const uint64_t h = payload_hash(body);
    atomic_file(dir + "/cmd." + std::to_string(cmd_idx),
                "ADMIT " + std::to_string(epoch) + " " + std::to_string(req_id) + " " + std::to_string(h) + "\n");
}

std::vector<int> parse_resp_tokens(const std::string& resp, bool& ended_ok) {
    std::vector<int> out;
    ended_ok = false;
    std::istringstream iss(resp);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.rfind("t ", 0) == 0) {
            std::istringstream ls(line.substr(2));
            int tok = 0;
            while (ls >> tok) out.push_back(tok);
        } else if (line.rfind("end length ", 0) == 0 || line.rfind("end stop ", 0) == 0) {
            ended_ok = true;
        }
    }
    return out;
}

struct RankBarrier {
    unsigned n;
    std::mutex mu;
    std::condition_variable cv;
    unsigned arrived = 0, gen = 0;
    explicit RankBarrier(unsigned nr) : n(nr) {}
    void arrive() {
        std::unique_lock<std::mutex> lk(mu);
        const unsigned g = gen;
        if (++arrived == n) { arrived = 0; ++gen; cv.notify_all(); }
        else cv.wait(lk, [&] { return gen != g; });
    }
};

void test_service_live_switch_mid_decode() {
    constexpr unsigned NR = 2;
    constexpr uint64_t EPOCH = 11;
    constexpr uint64_t GROUP_ID = 4242;
    constexpr uint32_t TARGET_NR = 3;
    const std::string root = mktemp_dir("gr8_live_sw_");
    std::vector<std::string> dirs(NR);
    for (unsigned r = 0; r < NR; ++r) {
        dirs[r] = root + "/r" + std::to_string(r);
        ::mkdir(dirs[r].c_str(), 0700);
    }

    int sp[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0, "socketpair");
    std::vector<ControlLinks> links(NR);
    links[0].links.push_back(std::make_unique<ControlIO>(sp[0]));
    links[1].links.push_back(std::make_unique<ControlIO>(sp[1]));

    RankBarrier bar(NR);
    std::vector<std::unique_ptr<RunPool>> pools(NR);
    std::atomic<int> switch_calls{0};
    std::atomic<int> decode_steps_r0{0};
    std::atomic<int> prefill_calls{0};
    std::atomic<int> retire_calls{0};
    std::vector<MockSwitchState> sw_states(NR);
    std::vector<std::unique_ptr<LiveSwitchCoordinator>> coords(NR);
    std::vector<std::unique_ptr<Service>> svcs(NR);

    for (unsigned r = 0; r < NR; ++r) {
        pools[r] = std::make_unique<RunPool>(64, 64, 64 * 64);
        sw_states[r].active_nr = NR;
        coords[r] = std::make_unique<LiveSwitchCoordinator>(sw_states[r].ops());
        CHECK(coords[r]->prepare(SwitchPlan{GROUP_ID, NR, TARGET_NR, 0, 256, 1024}));

        BootstrapConfig cfg;
        cfg.rank = r;
        cfg.ranks = NR;
        cfg.head_ip = "127.0.0.1";
        cfg.port = 1;
        cfg.directory = dirs[r];
        cfg.runtime_signature = "live-switch-test";
        cfg.prefill_min = 4;
        cfg.scheduler.epoch = EPOCH;
        cfg.scheduler.pool_rows = 64 * 64;
        cfg.scheduler.max_rows = 24;
        cfg.scheduler.max_sessions = 4;
        cfg.scheduler.mtp = 1;
        cfg.scheduler.guard_tokens = 4;
        cfg.scheduler.prefill_rows = 256;
        cfg.scheduler.two_lanes = false;
        cfg.scheduler.notice_max = 4;

        NativeHooks nh;
        nh.pool.pool = pools[r].get();
        nh.bind = [](const Grant&) {};
        nh.retire = [&](const Slot&) { retire_calls.fetch_add(1); };
        nh.fence = []() {};
        nh.switch_width = [&, r](const SwitchOp& sw) {
            CHECK(sw.group_id == GROUP_ID && sw.target_nr == TARGET_NR);
            coords[r]->cutover_at_fence(sw);
            coords[r]->finish_and_defrag();
            switch_calls.fetch_add(1);
        };
        nh.dispatch.bucket = [](bool, int) { return 0; };
        nh.dispatch.mode = [](int, int) {};
        nh.dispatch.prefill = [&](const Slot&, const Payload&, uint32_t, uint32_t) {
            prefill_calls.fetch_add(1);
            bar.arrive();
        };
        nh.dispatch.launch = [&, r](const Launch& l) {
            std::vector<int> out;
            out.reserve(l.step.r.size());
            for (const auto& row : l.step.r) {
                // Deterministic token strictly a function of (row.tok, row.pos) regardless of width.
                out.push_back(((row.tok * 3 + row.pos + 7) % 900) + 10);
            }
            if (r == 0 && l.step.scheduler_decode) {
                decode_steps_r0.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            bar.arrive();
            return out;
        };
        svcs[r] = std::make_unique<Service>(cfg, links[r], std::move(nh), /*chunk=*/4);
        svcs[r]->quit_linger_ms = 500;
    }

    std::vector<std::thread> threads;
    std::mutex err_mu;
    std::string thread_err;
    for (unsigned r = 0; r < NR; ++r) {
        threads.emplace_back([&, r] {
            try { svcs[r]->run(); }
            catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!thread_err.empty()) thread_err += "; ";
                thread_err += "rank " + std::to_string(r) + ": " + e.what();
            }
        });
    }

    // Wait for ready.s2
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!exists(dirs[0] + "/ready.s2") && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(exists(dirs[0] + "/ready.s2"));

    // Admit 2 concurrent requests (each 24 tokens so both are mid-decode when SWITCH arrives; Ingress starts at cmd.0)
    write_req_and_admit(dirs[0], EPOCH, 0, 1, {11, 22, 33, 44}, 24);
    write_req_and_admit(dirs[0], EPOCH, 1, 2, {55, 66, 77, 88}, 24);

    // Wait until rank 0 has run several decode steps, then inject SWITCH mid-decode
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (decode_steps_r0.load() < 4 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(decode_steps_r0.load() >= 4);

    atomic_file(dirs[0] + "/cmd.2",
                "SWITCH " + std::to_string(EPOCH) + " " + std::to_string(GROUP_ID) + " " + std::to_string(TARGET_NR) + "\n");

    // Wait for both responses to finish cleanly
    bool r1_done = false, r2_done = false;
    std::vector<int> t1, t2;
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (!(r1_done && r2_done) && std::chrono::steady_clock::now() < deadline) {
        t1 = parse_resp_tokens(slurp(dirs[0] + "/resp.1"), r1_done);
        t2 = parse_resp_tokens(slurp(dirs[0] + "/resp.2"), r2_done);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    atomic_file(dirs[0] + "/quit", "");
    for (auto& t : threads) t.join();

    CHECK(thread_err.empty());
    if (!thread_err.empty()) std::fprintf(stderr, "thread_err: %s\n", thread_err.c_str());
    CHECK(switch_calls.load() == int(NR));
    for (unsigned r = 0; r < NR; ++r) {
        CHECK(sw_states[r].active_nr == TARGET_NR);
        CHECK(sw_states[r].active_group == GROUP_ID);
        CHECK(coords[r]->phase() == SwitchPhase::Completed);
        CHECK(coords[r]->stats().defrag_armed);
    }
    CHECK(r1_done && t1.size() == 24);
    CHECK(r2_done && t2.size() == 24);
    // Verify exact deterministic tokens across the mid-decode width switch
    auto expected_tokens = [](int last_prompt, int prompt_len, int n_gen) {
        std::vector<int> out;
        int cur = last_prompt;
        for (int i = 0; i < n_gen; ++i) {
            cur = ((cur * 3 + (prompt_len - 1 + i) + 7) % 900) + 10;
            out.push_back(cur);
        }
        return out;
    };
    CHECK(t1 == expected_tokens(44, 4, 24));
    CHECK(t2 == expected_tokens(88, 4, 24));
    // Zero extra prefill replays after initial admission
    CHECK(prefill_calls.load() == int(NR) * 2);
}

} // namespace

int main() {
    test_coordinator_precapture_happy_path();
    test_coordinator_deferred_capture_path();
    test_coordinator_rollbacks();
    test_service_live_switch_mid_decode();
    if (!g_fails) std::printf("live_switch_test: PASS\n");
    return g_fails ? 1 : 0;
}
