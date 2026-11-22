// Host-only tests for EL7 (#155) in-process width-module lifecycle and per-layout CUDA graph capture policy:
//  1. choose_capture_mode boundary checks (CaptureAhead iff MemAvailable > Y + GR8_MEM_MIN, overflow safe).
//  2. CaptureAhead transition (4 -> 3 -> 6): target graphs capture before cutover (0 ms capture inside cutover),
//     old graphs release and old width module is destroyed cleanly.
//  3. CaptureAfter transition (tight memory, e.g. Flash 2 -> 1): skips capture-ahead, releases old graphs FIRST
//     inside cutover (returning Y_old to MemAvailable), then captures target graphs.
//  4. Mid-stage failure & rollback_staged(): capture-ahead failure or explicit rollback frees staged graphs &
//     destroys staged module cleanly with 0 leaks while active width stays untouched.
//  5. Refusals: unsupported width, duplicate staging, cutover without staged target.
// Zero CUDA dependency; runs under ASan+UBSan in scripts/host_tests.sh.
#include "../width_module.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace g6rt;

namespace {
int g_fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fails; } } while (0)

struct SimMem {
    uint64_t avail = 0;
    int live_modules = 0;
    int live_graphs = 0;
    bool fail_next_capture = false;
    std::vector<std::string> log;
};

struct FakeMod {
    SimMem* mem = nullptr;
    int nr = 0;
    uint64_t y_bytes = 0;
    bool captured = false;
};

uint64_t sim_avail(void* ctx) { return static_cast<SimMem*>(ctx)->avail; }

int fake_capture(void* ctx, int ahead, uint64_t* gb, double* ms, char* err, size_t errlen) {
    auto* m = static_cast<FakeMod*>(ctx);
    m->mem->log.push_back("capture_tp" + std::to_string(m->nr) + (ahead ? "_ahead" : "_now"));
    if (m->mem->fail_next_capture) {
        m->mem->fail_next_capture = false;
        if (err && errlen) std::snprintf(err, errlen, "simulated capture OOM on TP=%d", m->nr);
        return 1;
    }
    if (m->mem->avail < m->y_bytes) {
        if (err && errlen) std::snprintf(err, errlen, "insufficient memory for TP=%d graphs", m->nr);
        return 1;
    }
    m->mem->avail -= m->y_bytes;
    m->captured = true;
    ++m->mem->live_graphs;
    if (gb) *gb = m->y_bytes;
    if (ms) *ms = 270.0;
    return 0;
}

void fake_release(void* ctx) {
    auto* m = static_cast<FakeMod*>(ctx);
    if (!m->captured) return;
    m->mem->log.push_back("release_graphs_tp" + std::to_string(m->nr));
    m->mem->avail += m->y_bytes;
    m->captured = false;
    --m->mem->live_graphs;
}

void fake_destroy(void* ctx) {
    auto* m = static_cast<FakeMod*>(ctx);
    m->mem->log.push_back("destroy_tp" + std::to_string(m->nr));
    --m->mem->live_modules;
    delete m;
}

int fake_create(const G6HostRuntime* host, const G6WidthSpec* spec, G6WidthModule* out, char*, size_t) {
    auto* mem = static_cast<SimMem*>(host->mem_ctx);
    auto* m = new FakeMod{mem, spec->nr, spec->est_graph_bytes, false};
    ++mem->live_modules;
    mem->log.push_back("create_tp" + std::to_string(spec->nr));
    out->nr = spec->nr;
    out->ctx = m;
    out->capture_graphs = fake_capture;
    out->release_graphs = fake_release;
    out->destroy = fake_destroy;
    return 0;
}

void test_capture_policy() {
    constexpr uint64_t GB = 1ull << 30;
    CHECK(choose_capture_mode(10 * GB, 4 * GB, 4 * GB) == CaptureMode::CaptureAhead);
    CHECK(choose_capture_mode(8 * GB, 4 * GB, 4 * GB) == CaptureMode::CaptureAfter);
    CHECK(choose_capture_mode(5 * GB, 4 * GB, 4 * GB) == CaptureMode::CaptureAfter);
    CHECK(choose_capture_mode(UINT64_MAX, UINT64_MAX - 10, 100) == CaptureMode::CaptureAfter);
}

void test_capture_ahead_and_after_and_rollback() {
    constexpr uint64_t GB = 1ull << 30;
    WidthRegistry reg;
    reg.add(1, fake_create);
    reg.add(2, fake_create);
    reg.add(3, fake_create);
    reg.add(4, fake_create);
    reg.add(6, fake_create);
    CHECK(reg.has(4) && reg.has(3) && reg.has(6) && !reg.has(5));

    SimMem mem;
    mem.avail = 20 * GB;
    G6HostRuntime host{};
    host.mem_min_bytes = 6 * GB;
    host.mem_avail_bytes = sim_avail;
    host.mem_ctx = &mem;

    std::string err;
    {
        WidthCoordinator coord(reg, host);
        // Boot at TP=4 (graph memory Y = 3 GB).
        CHECK(coord.activate(G6WidthSpec{4, 0, 1, nullptr, 3 * GB}, err));
        CHECK(coord.active_nr() == 4 && mem.live_modules == 1 && mem.live_graphs == 1);
        CHECK(mem.avail == 17 * GB);

        // 1. CaptureAhead 4 -> 3 (avail 17 GB > 3 GB + 6 GB): target captures BEFORE cutover!
        CaptureMode mode = CaptureMode::CaptureAfter;
        CHECK(coord.stage_target(G6WidthSpec{3, 0, 2, nullptr, 3 * GB}, mode, err));
        CHECK(mode == CaptureMode::CaptureAhead && coord.staged_captured());
        CHECK(mem.live_modules == 2 && mem.live_graphs == 2 && mem.avail == 14 * GB);

        CutoverReport rep;
        CHECK(coord.cutover(rep, err));
        CHECK(rep.old_nr == 4 && rep.new_nr == 3 && rep.captured_ahead && !rep.captured_in_cutover);
        CHECK(coord.active_nr() == 3 && !coord.has_staged());
        CHECK(mem.live_modules == 1 && mem.live_graphs == 1 && mem.avail == 17 * GB);

        // 2. CaptureAfter 3 -> 1 under tight memory: set avail = 4 GB (<= Y=3 GB + min=6 GB),
        //    so staging MUST NOT precapture; cutover MUST release TP=3 graphs FIRST (avail 4->7 GB) then capture TP=1!
        mem.avail = 4 * GB;
        mem.log.clear();
        CHECK(coord.stage_target(G6WidthSpec{1, 0, 3, nullptr, 5 * GB}, mode, err));
        CHECK(mode == CaptureMode::CaptureAfter && !coord.staged_captured());
        CHECK(mem.live_modules == 2 && mem.live_graphs == 1 && mem.avail == 4 * GB);

        CHECK(coord.cutover(rep, err));
        CHECK(rep.old_nr == 3 && rep.new_nr == 1 && !rep.captured_ahead && rep.captured_in_cutover);
        CHECK(rep.target_graph_bytes == 5 * GB);
        CHECK(coord.active_nr() == 1 && mem.live_modules == 1 && mem.live_graphs == 1);
        CHECK(mem.avail == 2 * GB); // 4 + 3 (old TP=3 freed) - 5 (new TP=1) = 2 GB
        // Verify exact ordering: release_graphs_tp3 ran BEFORE capture_tp1_now!
        size_t pos_rel = 999, pos_cap = 999;
        for (size_t i = 0; i < mem.log.size(); ++i) {
            if (mem.log[i] == "release_graphs_tp3") pos_rel = i;
            if (mem.log[i] == "capture_tp1_now") pos_cap = i;
        }
        CHECK(pos_rel < pos_cap && pos_cap < 999);

        // 3. Mid-stage capture-ahead failure -> clean rollback, active TP=1 untouched, 0 leaked modules/graphs!
        mem.avail = 20 * GB;
        mem.fail_next_capture = true;
        CHECK(!coord.stage_target(G6WidthSpec{6, 0, 4, nullptr, 3 * GB}, mode, err));
        CHECK(!coord.has_staged() && coord.active_nr() == 1);
        CHECK(mem.live_modules == 1 && mem.live_graphs == 1 && mem.avail == 20 * GB);

        // 4. Explicit rollback_staged() after successful CaptureAhead -> frees staged graphs & module cleanly!
        CHECK(coord.stage_target(G6WidthSpec{6, 0, 5, nullptr, 3 * GB}, mode, err));
        CHECK(coord.has_staged() && coord.staged_captured() && mem.avail == 17 * GB);
        coord.rollback_staged();
        CHECK(!coord.has_staged() && coord.active_nr() == 1);
        CHECK(mem.live_modules == 1 && mem.live_graphs == 1 && mem.avail == 20 * GB);

        // 5. Refusals: unlinked width TP=5, cutover without staged target.
        CHECK(!coord.stage_target(G6WidthSpec{5, 0, 6, nullptr, 1 * GB}, mode, err));
        CHECK(!coord.cutover(rep, err));
    }
    // Destructor must cleanly release all remaining modules and graphs!
    CHECK(mem.live_modules == 0 && mem.live_graphs == 0);
}
} // namespace

int main() {
    test_capture_policy();
    test_capture_ahead_and_after_and_rollback();
    std::printf(g_fails ? "width_module_test: %d FAIL\n" : "width_module_test: PASS\n", g_fails);
    return g_fails != 0;
}
