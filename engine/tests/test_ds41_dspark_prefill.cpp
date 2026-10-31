// Host test of the DSpark prefill contract (engine/runtime/ds41_dspark_prefill.h, docs/DEEPSEEK-DSPARK.md section 9): no CUDA. Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I engine engine/tests/test_ds41_dspark_prefill.cpp -o /tmp/test_ds41_dspark_prefill
// Checks (a) the GR8_DS_PREFILL_KEEP knob, (b) the non-final chunk plan: DS_MAIN present, finish / head / step_out gone, publishes c = 0..79 kept,
// (c) the chunk -> row range -> subchunk arithmetic at W = 8 / 32 / 128 / 256 / 1024 and the tap / DS_MAIN positions across every chunk boundary,
// (d) a model of one DSpark layer ring (136 slots, slot = pos % 136): after the prompt AND after decode rounds, every key a draft block reads
// (q-127 .. q) is present with its own position, with and without pruning, (e) falsifiers: each rule broken on purpose must be caught.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "runtime/ds41_dspark_prefill.h"
#include "runtime/ds41_coll_order.h"

using namespace ds41;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

static dsp::Knobs knobs(std::map<std::string, std::string> m)
{
    return dsp::parse_knobs([&](const char* n) -> const char* { auto it = m.find(n); return it == m.end() ? nullptr : it->second.c_str(); });
}
static std::vector<int> iota(int n) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = i; return v; }

// ---- a model of one DSpark layer ring: slot = pos % 136, content = the position of the key written last
struct Ring
{
    std::vector<int> slot = std::vector<int>(dsp::RING, -1);
    void write(int pos) { slot[pos % dsp::RING] = pos; }
    // the main keys of a draft block with last accepted position q: q-127 .. q (the 5 block keys are written by the draft layer itself)
    bool window_ok(int q) const
    {
        for (int p = std::max(0, q - (geom::WINDOW - 1)); p <= q; ++p) if (slot[p % dsp::RING] != p) return false;
        return true;
    }
};
struct Sim
{
    Ring ring;
    std::vector<int> order;     // positions in launch order
    bool race = false;          // one launch held two rows with the same slot
    int chunks_with_main = 0, chunks_without = 0;
};
// One prompt of N tokens: hook chunks cover positions 0 .. N-2 (width W), then the tail N-1 as a decode-step DS_MAIN (one row).
// cap: rows per ST_DS_MAIN launch. reverse / drop: falsifiers (launch order reversed, one position never written).
static Sim run_prompt(int W, int N, int keep, int cap, bool reverse = false, int drop = -1)
{
    Sim s;
    auto launch = [&](const std::vector<int>& rows) {
        std::set<int> slots;
        for (int p : rows) if (!slots.insert(p % dsp::RING).second) s.race = true;
        for (int p : rows) { if (p == drop) continue; s.ring.write(p); s.order.push_back(p); }
    };
    std::vector<std::vector<int>> launches;
    if (N > 1)
        for (const auto& c : dsp::pf::chunks(0, N - 1, W))
        {
            const dsp::pf::Range r = dsp::pf::main_rows(c.first, c.rows, N, keep);
            (r.empty() ? s.chunks_without : s.chunks_with_main) += 1;
            for (const auto& u : dsp::pf::subchunks(r, cap))
            {
                std::vector<int> rows;
                for (int i = 0; i < u.rows; ++i) rows.push_back(c.first + u.row0 + i);
                launches.push_back(rows);
            }
        }
    launches.push_back({N - 1});   // PromptTail: the main step's own DS_MAIN
    if (reverse) std::reverse(launches.begin(), launches.end());
    for (const auto& l : launches) launch(l);
    return s;
}
// decode rounds after the prompt: the draft block reads at q, then the verify step writes rows P .. P+k (P = q+1), accept a drafts, q' = P + a.
// The block's own 5 keys are written by the draft layer before it reads (positions q+1 .. q+5): they share slots with the verify rows to come.
// Returns false at the first round whose window is not complete. Both rings (pruned and full) get the same writes.
static bool run_rounds(Ring& r, int N, int k, int rounds)
{
    int q = N - 1;
    for (int t = 0; t < rounds; ++t)
    {
        if (!r.window_ok(q)) return false;
        for (int p = q + 1; p <= q + 5; ++p) r.write(p);                 // the block's own keys
        const int P = q + 1;
        for (int p = P; p <= P + k; ++p) r.write(p);                      // ST_DS_MAIN of the verify rows (rejected rows included)
        const int a = t % (k + 1);                                        // 0 .. k accepted, all values in turn
        q = P + a;
    }
    return r.window_ok(q);
}

int main()
{
    // (a) the knob
    {
        EXPECT(knobs({}).err.empty() && knobs({}).prefill_keep == 0);
        EXPECT(knobs({{"GR8_DS_PREFILL_KEEP", "0"}}).err.empty() && knobs({{"GR8_DS_PREFILL_KEEP", "128"}}).prefill_keep == 128);
        EXPECT(knobs({{"GR8_DS_PREFILL_KEEP", "4096"}}).err.empty() && knobs({{"GR8_DS_PREFILL_KEEP", "4096"}}).prefill_keep == 4096);
        for (const char* bad : {"127", "1", "-1", "x", "128.5"}) EXPECT(!knobs({{"GR8_DS_PREFILL_KEEP", bad}}).err.empty());   // below the window: refused, not clamped
        EXPECT(dsp::pf::keep_valid(0) && dsp::pf::keep_valid(128) && !dsp::pf::keep_valid(127) && !dsp::pf::keep_valid(-1));
    }
    // (b) the non-final chunk plan, 40 layers, Engram off and on
    for (const bool engram : {false, true})
    {
        const auto ids = iota(40);
        const auto verify = plan(ids, 0, 40, true, true, engram, true);
        const auto nf = dsp::pf::nonfinal_plan(verify, true), nf0 = dsp::pf::nonfinal_plan(verify, false);
        EXPECT(dsp::pf::check_nonfinal_plan(verify, nf, true) == "");
        EXPECT(dsp::pf::check_nonfinal_plan(verify, nf0, false) == "");
        EXPECT(dsp::pf::count_stage(nf, ST_DS_TAP) == 3 && dsp::pf::count_stage(nf, ST_DS_MAIN) == 1 && nf.back().stage == ST_DS_MAIN);
        EXPECT(dsp::pf::count_stage(nf, ST_FINISH) == 0 && dsp::pf::count_stage(nf, ST_HEAD_LOCAL) == 0 && dsp::pf::count_stage(nf, ST_HEAD_MERGE) == 0 &&
               dsp::pf::count_stage(nf, ST_STEP_OUT) == 0);
        EXPECT(dsp::pf::count_stage(nf0, ST_DS_TAP) == 0 && dsp::pf::count_stage(nf0, ST_DS_MAIN) == 0);
        EXPECT(nf.size() == verify.size() - 5 && nf0.size() == verify.size() - 5 - 4);   // step_in + finish + head_local + head_merge + step_out; 3 taps + main
        EXPECT(dsp::pf::count_stage(nf, ST_STEP_IN) == 0 && dsp::pf::count_stage(nf0, ST_STEP_IN) == 0 && verify[0].stage == ST_STEP_IN);   // the decode plan has it, the chunk does not
        EXPECT(nf[0].stage == ST_EMBED && nf0[0].stage == ST_EMBED && dsp::pf::count_stage(nf, ST_EMBED) == 1);                          // the prologue embed stays, first
        EXPECT(dsp::pf::publishes(verify) == 81 && dsp::pf::publishes(nf) == 80 && dsp::pf::publishes(nf0) == 80);   // c = 0..79 stay, c = 80 is the head merge
        EXPECT(dsp::pf::count_stage(nf, ST_ENGRAM) == (engram ? 2 : 0));
        // B9's collective order of a step that skips the head leaves out c = 2*nl and nothing else (Engram on: 81, 82 stay)
        const auto ord = coll_exec_order(ids, engram ? 2 : 0, false);
        if (engram)
        {
            std::set<int> s(ord.begin(), ord.end());
            EXPECT(ord.size() == 82 && s.size() == 82 && !s.count(80) && s.count(79) && s.count(81) && s.count(82));
        }
        else EXPECT(ord.empty());
        // the tail row (PromptTail) is a decode step: its plan is the decode plan itself, finish + head + c = 80 included
        EXPECT(dsp::pf::count_stage(verify, ST_FINISH) == 1 && verify[verify.size() - 5].stage == ST_FINISH && verify[verify.size() - 4].stage == ST_DS_MAIN);
        EXPECT(dsp::pf::count_stage(verify, ST_STEP_IN) == 1);                                                                              // and its StepIn
    }
    // (b2) falsifiers of the plan check: each broken plan is refused
    {
        const auto verify = plan(iota(40), 0, 40, true, true, false, true);
        const auto good = dsp::pf::nonfinal_plan(verify, true);
        auto bad = [&](auto edit, bool dsp_rows = true) { auto p = good; edit(p); return dsp::pf::check_nonfinal_plan(verify, p, dsp_rows) != ""; };
        auto erase_stage = [](std::vector<Op>& p, Stage s, int n = 0) {
            for (size_t x = 0; x < p.size(); ++x) if (p[x].stage == s && n-- == 0) { p.erase(p.begin() + (long)x); return; }
        };
        EXPECT(bad([&](std::vector<Op>& p) { erase_stage(p, ST_DS_MAIN); }));                                              // DS_MAIN missing
        EXPECT(bad([&](std::vector<Op>& p) { erase_stage(p, ST_DS_TAP, 1); }));                                            // layer 38's tap missing
        EXPECT(bad([&](std::vector<Op>& p) { erase_stage(p, ST_PUBLISH_FFN, 7); }));                                       // a layer publish lost
        EXPECT(bad([&](std::vector<Op>& p) { Op f; f.stage = ST_FINISH; f.unit = U_CONS_F; p.insert(p.end() - 1, f); }));  // finish kept
        EXPECT(bad([&](std::vector<Op>& p) { Op h; h.stage = ST_HEAD_MERGE; h.unit = U_HEAD_MERGE; p.push_back(h); }));    // c = 80 published
        EXPECT(bad([&](std::vector<Op>& p) { std::swap(p[p.size() - 1], p[p.size() - 2]); }));                             // DS_MAIN before the last publish_ffn
        EXPECT(bad([&](std::vector<Op>& p) { for (Op& o : p) if (o.stage == ST_DS_TAP && o.l == 39) o.l = 36; }));         // a tap outside 37..39
        EXPECT(bad([&](std::vector<Op>& p) { Op s; s.stage = ST_STEP_IN; s.unit = U_STEP_IN; p.insert(p.begin(), s); }));   // ST_STEP_IN restored at the front: it would overwrite the first MX rows of tok / pos / tgt
        EXPECT(bad([&](std::vector<Op>& p) { Op s; s.stage = ST_STEP_IN; s.unit = U_STEP_IN; p.insert(p.begin() + 1, s); }));   // ... or anywhere else
        EXPECT(bad([&](std::vector<Op>& p) { erase_stage(p, ST_EMBED); }));                                                 // the prologue embed lost
        EXPECT(bad([&](std::vector<Op>& p) { (void)p; }, false));                                                            // a chunk with no kept row must NOT carry DSpark stages
        EXPECT(!dsp::pf::nonfinal_plan(verify, false).empty() && dsp::pf::check_nonfinal_plan(verify, dsp::pf::nonfinal_plan(verify, false), true) != "");
    }
    // (c) arithmetic at every chunk width: the rows of a chunk, the launches, and the positions across each boundary
    for (const int W : {8, 32, 128, 256, 1024})
        for (const int N : {1, 2, 3, 128, 129, 130, 1025, 1100, 4097, 8192})
            for (const int keep : {0, 128, 129, 136, 192, 1024})
            {
                const int p0 = dsp::pf::keep_first_pos(N, keep);
                std::vector<int> written;                                 // DS_MAIN positions in launch order over the hook's chunks
                int prev_end = -1;                                        // last position of the previous chunk (all rows)
                if (N > 1)
                    for (const auto& c : dsp::pf::chunks(0, N - 1, W))
                    {
                        EXPECT(c.first == prev_end + 1 && c.rows >= 1 && c.rows <= W);   // chunks tile 0 .. N-2: no gap, no overlap, last one partial
                        prev_end = c.first + c.rows - 1;
                        const auto r = dsp::pf::main_rows(c.first, c.rows, N, keep);
                        EXPECT(r.hi == c.rows && r.lo >= 0 && r.lo <= r.hi);
                        if (!r.empty()) { EXPECT(c.first + r.lo >= p0); EXPECT(r.lo == 0 || c.first + r.lo - 1 < p0); }   // exactly the suffix >= p0
                        else EXPECT(c.first + c.rows - 1 < p0);                                                            // a skipped chunk is entirely older
                        for (const int cap : {8, 32, dsp::pf::cap_for(1024)})
                        {
                            int row = r.lo, nsub = 0;
                            for (const auto& u : dsp::pf::subchunks(r, cap))
                            {
                                EXPECT(u.row0 == row && u.rows >= 1 && u.rows <= cap && u.rows <= dsp::RING);   // contiguous, within the cap, within the ring
                                row += u.rows; ++nsub;
                            }
                            EXPECT(row == r.hi || r.empty());
                            EXPECT(nsub == (r.empty() ? 0 : (r.hi - r.lo + cap - 1) / cap));
                        }
                        for (int i = r.lo; i < r.hi; ++i) written.push_back(c.first + i);
                    }
                if (N > 1) EXPECT(prev_end == N - 2);                    // the hook never touches the tail token N-1
                // tap / DS_MAIN positions: ascending, consecutive, from p0 (or 1.. when pruned past the prompt start) to N-2
                for (size_t x = 1; x < written.size(); ++x) EXPECT(written[x] == written[x - 1] + 1);
                if (N > 1) { EXPECT(!written.empty() || p0 >= N - 1); if (!written.empty()) EXPECT(written.front() == std::min(p0, N - 2) + (p0 > N - 2 ? 1 : 0) || written.front() == p0); }
                EXPECT((int)written.size() == std::max(0, (N - 1) - std::min(p0, N - 1)));
            }
    // (c2) the boundary values themselves: W = 128 / 256 / 1024, keep all: chunk c's first DS_MAIN row is position c*W, the previous chunk ends at c*W-1
    for (const int W : {128, 256, 1024})
    {
        const int N = 3 * W + 50;
        const auto cs = dsp::pf::chunks(0, N - 1, W);
        EXPECT(cs.size() == 4 && cs[0].first == 0 && cs[1].first == W && cs[2].first == 2 * W && cs[3].first == 3 * W && cs[3].rows == 49);
        for (size_t c = 1; c < cs.size(); ++c)
        {
            const auto prev = dsp::pf::subchunks(dsp::pf::main_rows(cs[c - 1].first, cs[c - 1].rows, N, 0), 32).back();
            const auto cur = dsp::pf::subchunks(dsp::pf::main_rows(cs[c].first, cs[c].rows, N, 0), 32).front();
            EXPECT(cs[c - 1].first + prev.row0 + prev.rows - 1 == (int)c * W - 1 && cs[c].first + cur.row0 == (int)c * W);
        }
        // pruned, the kept window starts inside chunk 3 for keep 128: p0 = N - 128 = 3W - 78 -> chunk 2 (rows 2W .. 3W-1) holds the start
        const int p0 = dsp::pf::keep_first_pos(N, 128);
        EXPECT(p0 == 3 * W - 78);
        const auto r2 = dsp::pf::main_rows(cs[2].first, cs[2].rows, N, 128);
        EXPECT(cs[2].first + r2.lo == p0 && r2.hi == W && dsp::pf::main_rows(cs[1].first, cs[1].rows, N, 128).empty() == (cs[1].first + cs[1].rows - 1 < p0));
        EXPECT(!dsp::pf::main_rows(cs[3].first, cs[3].rows, N, 128).empty() && dsp::pf::main_rows(cs[3].first, cs[3].rows, N, 128).lo == 0);
    }
    // (c3) the exact numbers of one prompt: W = 256, N = 1000, keep 128 -> p0 = 872, only chunk 3 (768 .. 998) has rows: 104 .. 230, 127 rows, + the tail
    {
        const auto cs = dsp::pf::chunks(0, 999, 256);
        EXPECT(cs.size() == 4 && cs[3].first == 768 && cs[3].rows == 231);
        EXPECT(dsp::pf::keep_first_pos(1000, 128) == 872);
        for (int c = 0; c < 3; ++c) EXPECT(dsp::pf::main_rows(cs[c].first, cs[c].rows, 1000, 128).empty());
        const auto r = dsp::pf::main_rows(768, 231, 1000, 128);
        EXPECT(r.lo == 104 && r.hi == 231 && r.hi - r.lo == 127);
        const auto s = dsp::pf::subchunks(r, 32);
        EXPECT(s.size() == 4 && s[0].row0 == 104 && s[0].rows == 32 && s[3].row0 == 200 && s[3].rows == 31);
        EXPECT(dsp::pf::subchunks(r, dsp::pf::cap_for(1024)).size() == 1);   // the pruned window fits ONE launch of <= 136 rows
        EXPECT(dsp::pf::tap_bytes(1024) == 31457280 && dsp::pf::tap_bytes(8) == 245760);
    }
    // (d) the ring: every combination of width, prompt length, keep, cap leaves a complete first-round window, and stays complete over decode rounds
    long sims = 0;
    for (const int W : {8, 32, 128, 256, 1024})
        for (const int N : {1, 2, 3, 100, 127, 128, 129, 136, 137, 257, 1000, 1025, 4097, 8192})
            for (const int keep : {0, 128, 129, 136, 192, 1024, 5000})
                for (const int cap : {8, 32, 136})
                {
                    Sim full = run_prompt(W, N, 0, cap), pruned = run_prompt(W, N, keep, cap);
                    EXPECT(!full.race && !pruned.race);                                          // no launch holds two rows of one slot
                    EXPECT(full.ring.window_ok(N - 1) && pruned.ring.window_ok(N - 1));          // the first round (q = N-1) reads a complete window
                    for (size_t x = 1; x < pruned.order.size(); ++x) EXPECT(pruned.order[x] > pruned.order[x - 1]);   // ascending: the newest position wins a slot
                    EXPECT(pruned.order.back() == N - 1 && (int)full.order.size() == N);         // keep all = every position once; pruned still ends with the tail
                    EXPECT((int)pruned.order.size() == N - dsp::pf::keep_first_pos(N, keep));
                    for (const int k : {1, 5})
                    {
                        Ring a = full.ring, b = pruned.ring;
                        EXPECT(run_rounds(a, N, k, 40) && run_rounds(b, N, k, 40));              // pruned = full on everything a draft reads, 40 rounds, every accept length
                    }
                    ++sims;
                }
    // (e) falsifiers: each rule broken on purpose is caught
    {
        const int W = 256, N = 1000;
        // e1 keep one row too few (127): position N-128 is never written, the first window is incomplete (needs the clamp the knob enforces)
        EXPECT(!run_prompt(W, N, 127, 32).ring.window_ok(N - 1));
        EXPECT(run_prompt(W, N, 128, 32).ring.window_ok(N - 1));
        // e2 drop one position inside the window, in the middle of a chunk, and the first (tail) position
        for (const int d : {N - 128, N - 100, 900, N - 2, N - 1}) EXPECT(!run_prompt(W, N, 128, 32, false, d).ring.window_ok(N - 1));
        // e3 a position OUTSIDE the window may be dropped (that is the whole saving): the full run without position 5 still has a complete window
        EXPECT(run_prompt(W, N, 0, 32, false, 5).ring.window_ok(N - 1) && !run_prompt(W, 100, 0, 32, false, 5).ring.window_ok(99));
        // e4 launches in descending order: the OLDEST position wins each slot, the window is wrong (slot = pos % 136 needs ascending launches)
        EXPECT(!run_prompt(W, N, 0, 32, true).ring.window_ok(N - 1));
        // e5 a launch wider than the ring: two rows share a slot in ONE launch (a race on the device), caught by the simulator
        EXPECT(run_prompt(W, 5000, 0, 137).race && !run_prompt(W, 5000, 0, 136).race && dsp::pf::cap_for(1024) == 136 && dsp::pf::cap_for(32) == 32);
        // e6 wider keep never shrinks the window; prompts shorter than the window have nothing to prune
        EXPECT(dsp::pf::keep_first_pos(100, 128) == 0 && dsp::pf::keep_first_pos(129, 128) == 1 && dsp::pf::keep_first_pos(5000, 0) == 0);
    }
    // (f) MainS of 128 rows (B1 GO 8:0x AM PT): cap = min(128, RING) = 128. Launch counts, the GEMM weight-tile model (k_gemm_mxfp8 GBM = 64: a launch reads
    // the weights once per 64 rows), the ring over every width / length / keep at cap 128, and the falsifier that cap 136 would overrun the 128-row scratch.
    {
        EXPECT(dsp::pf::MS_ROWS == 128 && dsp::pf::cap_for(dsp::pf::MS_ROWS) == 128 && dsp::pf::cap_for(256) == dsp::RING);
        const int cap = dsp::pf::cap_for(dsp::pf::MS_ROWS);
        auto launches = [&](int rows, int c) { return (int)dsp::pf::subchunks({0, rows}, c).size(); };
        auto tiles = [&](int rows, int c) { int t = 0; for (const auto& u : dsp::pf::subchunks({0, rows}, c)) t += (u.rows + 63) / 64; return t; };
        EXPECT(launches(1024, 8) == 128 && launches(1024, 32) == 32 && launches(1024, cap) == 8);        // MX 8 / MX 32 / 128-row scratch, unpruned W = 1024
        EXPECT(tiles(1024, 8) == 128 && tiles(1024, 32) == 32 && tiles(1024, cap) == 16 && tiles(1024, cap) == 1024 / 64);   // 16 = the floor: one weight read per 64 rows
        EXPECT(launches(127, cap) == 1 && launches(128, cap) == 1 && launches(129, cap) == 2 && launches(1, cap) == 1);
        // keep = 128, every width: the kept rows are ONE launch per chunk they touch (at most two chunks), each <= 128 rows = the scratch
        for (const int W : {8, 32, 128, 256, 1024})
            for (const int N : {129, 200, 1000, 1025, 4097, 8192})
            {
                int nl = 0, rows = 0, maxrows = 0;
                for (const auto& c : dsp::pf::chunks(0, N - 1, W))
                    for (const auto& u : dsp::pf::subchunks(dsp::pf::main_rows(c.first, c.rows, N, 128), cap)) { ++nl; rows += u.rows; maxrows = std::max(maxrows, u.rows); }
                EXPECT(rows == 127 && maxrows <= dsp::pf::MS_ROWS);                                      // positions N-128 .. N-2 (N-1 is the tail step)
                EXPECT(nl == (W >= 128 ? ((N - 128) / W == (N - 2) / W ? 1 : 2) : (N - 2) / W - (N - 128) / W + 1));   // 1 launch per touched chunk (rows of one chunk <= 128 when W >= 128)
            }
        long sims128 = 0;
        for (const int W : {8, 32, 128, 256, 1024})
            for (const int N : {1, 2, 3, 100, 127, 128, 129, 136, 137, 257, 1000, 1025, 4097, 8192})
                for (const int keep : {0, 128, 129, 136, 192, 1024, 5000})
                {
                    Sim full = run_prompt(W, N, 0, cap), pruned = run_prompt(W, N, keep, cap);
                    EXPECT(!full.race && !pruned.race && full.ring.window_ok(N - 1) && pruned.ring.window_ok(N - 1));
                    for (size_t x = 1; x < pruned.order.size(); ++x) EXPECT(pruned.order[x] > pruned.order[x - 1]);
                    for (const int k : {1, 5}) { Ring a = full.ring, b = pruned.ring; EXPECT(run_rounds(a, N, k, 40) && run_rounds(b, N, k, 40)); }
                    ++sims128;
                }
        // cap 136 on a 128-row scratch: a launch of 136 rows exists (no ring race, but it overruns MainS): the simulator sees rows > MS_ROWS
        int maxrows136 = 0;
        for (const auto& u : dsp::pf::subchunks({0, 1024}, dsp::pf::cap_for(256))) maxrows136 = std::max(maxrows136, u.rows);
        EXPECT(maxrows136 == 136 && maxrows136 > dsp::pf::MS_ROWS);
        std::printf("test_ds41_dspark_prefill: cap128 case ok (%ld ring sims at cap 128; W=1024 unpruned: 128 / 32 / 8 launches at cap 8 / 32 / 128, weight tiles 128 / 32 / 16)\n", sims128);
    }
    std::printf("test_ds41_dspark_prefill: all checks passed (%ld ring sims)\n", sims);
    return 0;
}
