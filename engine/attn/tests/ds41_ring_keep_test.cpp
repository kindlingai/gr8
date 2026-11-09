// Host only (no CUDA): ds41a::ring_keep, the window-ring retention rule of k_kv_prep (ds41_attn.cuh). Replays one launch
// into per-slot rings of R rows (slot = pos % R) and checks what the core's decode read (ds41_attn.cuh, positions
// tpos - 127 .. tpos of the row's own slot ring) needs: every such position is present, and no two kept rows of one
// launch write the same ring row of one slot (the kernel's blocks run in any order).
// A rule over the whole packed launch (row >= n - R) would write nothing for slots 0-2 of an 8 x 38 DS41 ragged pack at R 175.
// Build: g++ -std=c++17 -O2 -Wall -Wextra -I engine/attn engine/attn/tests/ds41_ring_keep_test.cpp
#include "ds41_ring_keep.h"
#include <cstdio>
#include <map>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); } } while (0)
constexpr int WIN = 128;

struct Launch { std::vector<int> slot, pos; };
// Segments of `rows` each on slots 0.., every segment a fresh prompt (first 0), as ds41::pack::error requires.
static Launch pack(int segs, int rows) {
    Launch l;
    for (int s = 0; s < segs; ++s) for (int r = 0; r < rows; ++r) { l.slot.push_back(s); l.pos.push_back(r); }
    return l;
}
// Uniform decode: `sess` sessions of `rps` rows (session = row / rps), each at context `ctx` (positions ctx .. ctx+rps-1).
static Launch uniform(int sess, int rps, int ctx) {
    Launch l;
    for (int s = 0; s < sess; ++s) for (int r = 0; r < rps; ++r) { l.slot.push_back(s); l.pos.push_back(ctx + r); }
    return l;
}

// Returns rows written per slot; checks presence and no same-slot ring-row collision.
static std::vector<int> run(const Launch& l, int R, bool use_sess, std::map<int, std::map<int, int>>* ring_out = nullptr) {
    const int n = int(l.slot.size());
    int nslots = 0; for (int s : l.slot) nslots = s + 1 > nslots ? s + 1 : nslots;
    std::vector<int> writes(size_t(nslots), 0);
    std::map<int, std::map<int, int>> ring;   // slot -> ring row -> position
    for (int row = 0; row < n; ++row) {
        if (!ds41a::ring_keep(row, n, R, use_sess ? l.slot.data() : nullptr)) continue;
        auto& r = ring[l.slot[size_t(row)]];
        CHECK(r.count(l.pos[size_t(row)] % R) == 0);   // a second write of one ring row in one launch = a race
        r[l.pos[size_t(row)] % R] = l.pos[size_t(row)];
        ++writes[size_t(l.slot[size_t(row)])];
    }
    if (ring_out) *ring_out = ring;
    return writes;
}

// Every row's decode window (its own slot, positions >= first position of its run, the earlier context is not in this
// launch) must be present in the slot ring after the launch.
static int missing_window(const Launch& l, int R) {
    std::map<int, std::map<int, int>> ring;
    run(l, R, true, &ring);
    std::map<int, int> first;
    for (size_t i = 0; i < l.slot.size(); ++i) if (!first.count(l.slot[i])) first[l.slot[i]] = l.pos[i];
    int miss = 0;
    for (size_t i = 0; i < l.slot.size(); ++i) {
        const int lo = l.pos[i] - (WIN - 1) > first[l.slot[i]] ? l.pos[i] - (WIN - 1) : first[l.slot[i]];
        for (int p = lo; p <= l.pos[i]; ++p) {
            auto& r = ring[l.slot[i]];
            if (!r.count(p % R) || r[p % R] != p) { ++miss; break; }
        }
    }
    return miss;
}

static void print(const char* what, const std::vector<int>& w) {
    std::printf("%s writes_by_slot=[", what);
    for (size_t i = 0; i < w.size(); ++i) std::printf("%s%d", i ? ", " : "", w[i]);
    std::printf("]\n");
}

int main() {
    // 8 x 38 and 2 x 38 packs at the compact S32 ring (175 = WIN + MX 48 - 1) and the wide ring 4223.
    for (int R : {175, 4223}) for (int segs : {8, 2}) {
        const Launch l = pack(segs, 38);
        const std::vector<int> w = run(l, R, true);
        char buf[64]; std::snprintf(buf, sizeof buf, "pack %dx38 R=%d", segs, R); print(buf, w);
        for (int x : w) CHECK(x == 38);
        CHECK(missing_window(l, R) == 0);
    }
    // The pack bounds: 8 segments of 64 rows (ds41::pack MAX_SEGMENT_ROWS, MAX_ROWS 512) at R 175.
    { const Launch l = pack(8, 64); for (int x : run(l, 175, true)) CHECK(x == 64); CHECK(missing_window(l, 175) == 0); }
    // Uniform decode maps keep every row (rows per session <= R), as a whole-launch rule does while n <= R.
    { const Launch l = uniform(32, 1, 1000); for (int x : run(l, 175, true)) CHECK(x == 1); }
    { const Launch l = uniform(8, 6, 500); for (int x : run(l, 175, true)) CHECK(x == 6); CHECK(missing_window(l, 175) == 0); }
    // 32 sessions x 6 rows = 192 > 175: a whole-launch rule would drop sessions 0-2; per run all are kept.
    { const Launch l = uniform(32, 6, 500); for (int x : run(l, 175, true)) CHECK(x == 6); CHECK(missing_window(l, 175) == 0); }
    // One slot run longer than R keeps exactly its last R rows, without collisions.
    { const Launch l = pack(1, 300); const std::vector<int> w = run(l, 175, true); CHECK(w[0] == 175);
      for (int row = 0; row < 300; ++row) CHECK(ds41a::ring_keep(row, 300, 175, l.slot.data()) == (row >= 300 - 175)); }
    // No session map (one sequence): unchanged, the last R rows of the launch.
    for (int n : {1, 100, 175, 176, 4000}) for (int row = 0; row < n; ++row)
        CHECK(ds41a::ring_keep(row, n, 175, nullptr) == (row >= n - 175));
    std::printf("ds41_ring_keep_test: %s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
