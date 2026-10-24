#include "../runtime/s2/flash_ring.h"
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <random>
#include <set>
#include <vector>
namespace {
constexpr int MX = 64;
struct StepIn { int pos[MX] = {}, kda_src = 0, kslot[MX] = {}, krun[MX] = {}; };
template <class F> bool throws(F f) {
    try { f(); } catch (const std::runtime_error&) { return true; }
    return false;
}
// The carry mode of k_kda_core (engine/kda/math.h, Args::src) on symbolic states: a state is the positions applied
// to it. src[0] = kda_src, src[m] = krun[m] - 1 (k_step_in).
void carry(std::vector<std::vector<int>>& slot, const StepIn& si, int M) {
    int cur = -1;
    std::vector<int> reg;
    for (int m = 0; m < M; ++m) {
        const int s = si.kslot[m], src = m ? si.krun[m] - 1 : si.kda_src;
        if (s != cur) {
            if (cur >= 0) slot.at(cur) = reg;
            if (src >= 0) reg = slot.at(src);
            cur = s;
        }
        reg.push_back(si.pos[m]);
    }
    slot.at(cur) = reg;
}
void ring() {
    g6s2::FlashRing r;
    assert(r.source(31) == 0);
    r.record(31, 1, false); assert(r.source(32) == 0);
    // Each accepted length, including rejection of every draft and full acceptance.
    for (int k = 1; k <= 7; ++k) for (int a = 0; a <= k; ++a) {
        r.record(32, k + 1, true); assert(r.source(33 + a) == a);
    }
    // A forced-token step after partial acceptance, then back to verify.
    r.record(32, 5, true); assert(r.source(35) == 2);
    r.record(35, 1, false); assert(r.source(36) == 0);
    for (int p : {35, 37}) assert(throws([&] { r.source(p); }));
    r.reset(); assert(r.source(0) == 0);
}
// One session through FlashSlots is the single-session code it replaces (flash_native.cuh before several sessions):
// kda_src = ring.source(pos), kslot = r on a verify step and 0 otherwise, krun all zero.
void solo() {
    for (int K : {1, 3, 7}) {
        g6s2::FlashSlots ks(1, K + 1);
        g6s2::FlashRing ref;
        std::mt19937 rng(K);
        int pos = 40;
        for (int step = 0; step < 200; ++step) {
            const bool verify = step % 17 != 0;
            const int n = verify ? 1 + int(rng() % (K + 1)) : 1 + int(rng() % 9);
            StepIn si;
            for (int r = 0; r < n; ++r) si.pos[r] = pos + r;
            ks.inputs(si, {0}, {0}, {n}, {verify});
            assert(si.kda_src == ref.source(pos));
            for (int r = 0; r < n; ++r) assert(si.kslot[r] == (verify ? r : 0) && si.krun[r] == 0);
            ref.record(pos, n, verify);
            pos += verify ? 1 + int(rng() % n) : n;
        }
    }
}
// S sessions sharing steps: disjoint slots, each session's first row loads exactly its own accepted history, and the
// same sources as the session would see alone; reset touches only its session.
void sessions() {
    for (int K : {1, 3, 7}) for (int S : {2, 3, 8}) {
        const int per = K + 1;
        g6s2::FlashSlots ks(S, per);
        std::vector<g6s2::FlashRing> alone(S);
        std::vector<std::vector<int>> slot(size_t(S * per)), done(S);
        std::vector<int> pos(S);
        for (int q = 0; q < S; ++q) pos[q] = 100 * q;
        std::mt19937 rng(K * 10 + S);
        for (int step = 0; step < 300; ++step) {
            std::vector<int> sess, row0, n;
            std::vector<bool> verify;
            StepIn si;
            int M = 0;
            for (int q = 0; q < S; ++q) {
                if (rng() % 4 == 0) continue;              // not in this step (finished its round, or drafting)
                const bool v = rng() % 13 != 0;            // else a forced-token step
                const int rows = v ? 1 + int(rng() % per) : 1;
                if (M + rows > MX) break;
                sess.push_back(q); row0.push_back(M); n.push_back(rows); verify.push_back(v);
                for (int r = 0; r < rows; ++r) si.pos[M + r] = pos[q] + r;
                M += rows;
            }
            if (sess.empty()) continue;
            ks.inputs(si, sess, row0, n, verify);
            for (size_t j = 0; j < sess.size(); ++j) {
                const int q = sess[j], b = q * per, src = j ? si.krun[row0[j]] - 1 : si.kda_src;
                assert(src == b + alone[q].source(pos[q]) && slot[src] == done[q]);
                alone[q].record(pos[q], n[j], verify[j]);
                for (int r = 0; r < n[j]; ++r) {
                    const int s = si.kslot[row0[j] + r];
                    assert(s >= b && s < b + per && s == b + (verify[j] ? r : 0));
                    if (r) assert(si.krun[row0[j] + r] == 0);
                }
            }
            carry(slot, si, M);
            for (size_t j = 0; j < sess.size(); ++j) {
                const int q = sess[j], a = verify[j] ? int(rng() % n[j]) : n[j] - 1;   // rows the session keeps - 1
                for (int r = 0; r <= a; ++r) done[q].push_back(pos[q] + r);
                pos[q] += a + 1;
            }
            if (step % 50 == 49) {                          // a session ends; a new one starts on its slot
                const int q = int(rng() % S);
                ks.reset(q); alone[q].reset();
                for (int r = 0; r < per; ++r) slot[q * per + r].clear();
                done[q].clear(); pos[q] = 0;
            }
        }
    }
}
void errors() {
    g6s2::FlashSlots ks(2, 4);
    StepIn si;
    assert(throws([] { g6s2::FlashSlots(0, 4); }) && throws([] { g6s2::FlashSlots(2, 0); }));
    assert(throws([&] { ks.inputs(si, {2}, {0}, {1}, {true}); }));                 // no such session
    assert(throws([&] { ks.inputs(si, {0}, {0}, {5}, {true}); }));                 // verify wider than the ring
    assert(throws([&] { ks.inputs(si, {0, 1}, {0, 2}, {1, 1}, {true, true}); }));  // a gap between sessions
    assert(throws([&] { ks.inputs(si, {0}, {0}, {1}, {}); }));
    si.pos[0] = 10; ks.inputs(si, {0}, {0}, {2}, {true});
    si.pos[0] = 13; assert(throws([&] { ks.inputs(si, {0}, {0}, {1}, {true}); }));  // past the last step's rows
    assert(ks.base(1) == 4 && throws([&] { ks.base(-1); }));
}
// FlashSlots::step from an s2::Step layout and its sessions' event classes is inputs() with verify = not the prompt tail
struct Step { std::vector<unsigned> sess, row0, rows; };
constexpr int TAIL = 3;
void step() {
    g6s2::FlashSlots a(3, 4), b(3, 4);
    StepIn x, y;
    const Step sp{{0, 2}, {0, 4}, {4, 1}};
    for (int r = 0; r < 5; ++r) x.pos[r] = y.pos[r] = 20 + r;
    a.step(x, sp, {1, TAIL}, TAIL);
    b.inputs(y, {0, 2}, {0, 4}, {4, 1}, {true, false});
    assert(x.kda_src == y.kda_src && x.kda_src == 0 && x.krun[4] == y.krun[4] && x.krun[4] == 8 + 1);
    for (int r = 0; r < 5; ++r) assert(x.kslot[r] == y.kslot[r]);
    assert(x.kslot[3] == 3 && x.kslot[4] == 8);
    assert(throws([&] { a.step(x, sp, {1}, TAIL); }));               // an event per session
}
// The boot capture set: every main and draft width S2 can launch, and nothing past MX
void shapes() {
    for (int N : {1, 2, 8}) for (int K : {1, 2, 3}) for (bool adapt : {false, true}) for (int chunk : {1, 16}) {
        const auto f = g6s2::flash_shapes(N, K, chunk, adapt, 64);
        for (int M = 1; M <= chunk; ++M) assert(f.mains.count(M) && f.drafts.count(M));
        for (int S = 1; S <= N; ++S) for (int k = adapt ? 1 : K; k <= K; ++k) assert(f.mains.count(S * (k + 1)));
        for (int M : f.mains) {                                      // and no other width
            bool round = M <= chunk;
            for (int S = 1; S <= N; ++S) for (int k = adapt ? 1 : K; k <= K; ++k) round |= M == S * (k + 1);
            assert(round);
        }
        assert(int(f.drafts.size()) == std::max(chunk, N * (K + 1)) && *f.drafts.rbegin() == std::max(chunk, N * (K + 1)));
        assert(int(f.chains.size()) == (K >= 2 ? N : 0));
    }
    // the solo set: prompt tails and one round
    const auto f = g6s2::flash_shapes(1, 3, 4, false, 64);
    assert((f.mains == std::set<int>{1, 2, 3, 4}) && f.drafts.size() == 4 && (f.chains == std::set<int>{1}));
    assert(throws([] { g6s2::flash_shapes(8, 3, 16, false, 31); }));   // 8 x 4 rows > MX
    assert(!throws([] { g6s2::flash_shapes(8, 3, 16, false, 32); }));
    assert(throws([] { g6s2::flash_shapes(0, 3, 16, false, 32); }) && throws([] { g6s2::flash_shapes(2, 0, 16, false, 32); }) &&
           throws([] { g6s2::flash_shapes(2, 3, 33, false, 32); }));
}
}
int main() {
    ring(); solo(); sessions(); errors(); step(); shapes();
    puts("flash_ring: PASS (all accepted lengths, insertion, reset, invalid positions; FlashSlots: one session as before, "
         "sessions sharing steps in their own slots, reset, invalid layouts, the s2::Step form; the capture set)");
}
