// GLM-5.3-Flash, DFlash2 with session checkpoints (GR8_GLM53F_CACHE, #340): s2/flash_dfl_window.h decides which session
// owns the prefill taps and where its drafter context starts; s2_batch.h Batch (GR8_DFL_REUSE) extends it into the
// window from the drafter's ring or warms it up on MTP. Host only, a fake engine: the window's rules alone, then the
// window driving a Batch as the Flash serve's hooks do (g6m4.cu dfl_install), through an agent's turns that hit their
// checkpoints. Every committed token is the target's.
#include "s2/flash_dfl_window.h"
#include "s2/flash_checkpoint.h"
#include "s2_batch.h"
#include <algorithm>
#include <cstdio>
using namespace s2;
using g6s2::FlashDflWindow;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1
constexpr int WIN = DFL_WIN, GRID = 8192;                        // the drafter's window, the cold prefill grid (4096 x 2)

// the window's rules on their own
static void rules()
{
    for (int reuse : {0, 1})
    {
        FlashDflWindow w(2, WIN, reuse);
        // a cold prompt: the taps are its whole window, filled from lo
        w.bind(0, 9000, 0);
        CHECK(w.quantum(0, 1, 9000, 0) && w.lo == 8999 - WIN && w.hi == 8999);
        CHECK(!w.quantum(0, 1, 9000, 4096));            // a later quantum of the same request keeps the window
        CHECK(w.ok(0) && w.fills(0) && w.start(0) == 8999 - WIN);
        // a checkpoint hit at or below lo: the prefill still taps the whole window
        w.bind(1, 12000, GRID);
        CHECK(w.quantum(1, 2, 12000, GRID) && w.lo == 11999 - WIN);
        CHECK(w.ok(1) && w.start(1) == 11999 - WIN);
        CHECK(!w.ok(0));                                 // slot 1's prefill took the taps
        // a hit inside the window: the drafter only with the reuse, its context from the hit
        w.bind(0, 9000, GRID);
        CHECK(w.quantum(0, 3, 9000, GRID));
        CHECK(w.ok(0) == bool(reuse));
        CHECK(reuse ? w.fills(0) && w.start(0) == GRID : !w.fills(0) && w.start(0) == 8999);
        // a hit with every prompt row but the last: no prefill, the drafter from its prompt step (reuse only)
        w.bind(1, GRID + 1, GRID);
        CHECK(w.ok(1) == bool(reuse) && !w.fills(1) && w.start(1) == GRID);
        // a new grant on the owner's slot drops the taps it does not own
        w.bind(0, 100, 0);
        CHECK(!w.fills(0) && !w.ok(0));
        // a 1-token prompt drafts from its prompt step, with or without the reuse
        w.bind(0, 1, 0);
        CHECK(w.ok(0) && w.start(0) == 0);
    }
    bool threw = false;
    try { FlashDflWindow w(1, WIN, 3); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { FlashDflWindow w(1, WIN, 1); w.ok(1); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// the Flash serve's DFlash2 hooks over a fake engine: bind, the first prefill quantum, dfl_ok at the prompt tail
// (Driver: a session alone), dfl_start's fill
struct Serve
{
    FlashDflWindow w;
    std::vector<DflReq> reqs;
    std::vector<char> rounds;         // per decode round: 'm' (MTP verify) or 'd' (DFlash2, 8 rows)
    std::vector<int> got;
    int fill_from = -1, fill_to = -1; // the last dfl_start's fill rows
    uint64_t next_id = 1;
    explicit Serve(int reuse) : w(2, WIN, reuse) {}
    Hooks hooks()
    {
        Hooks h;
        h.run = [this](const Step& s) {
            std::vector<int> out;
            for (const Row& r : s.r) out.push_back(target(r.spos - 1));
            if (s.kind == Step::MAIN && s.M() > 1) rounds.push_back(s.M() == 8 ? 'd' : 'm');
            return out;
        };
        h.dfl_start = [this](int q) {
            fill_from = fill_to = -1;
            if (w.fills(q)) { fill_from = w.from; fill_to = w.hi; }
            return w.start(q);
        };
        h.dfl = [this](const DflReq& r) {
            reqs.push_back(r);
            DflDrafts o{};
            const int q0 = r.pos0 + r.rows;
            for (int i = 1; i < DFL_B; ++i) { o.tok[i] = target(q0 + i - 1); o.lp[i] = -0.1f; }
            return o;
        };
        h.emit = [this](int, const int* t, int k, int) { got.insert(got.end(), t, t + k); return false; };
        return h;
    }
    // a request on slot q whose checkpoint hit restored `reused` rows: bind, its prefill quanta from there (none when
    // the hit leaves only the last row), then its prompt tail and decode, alone
    void turn(Batch& b, int q, const std::vector<int>& prompt, uint32_t reused, int n_gen)
    {
        reqs.clear(); rounds.clear(); got.clear();
        const uint32_t n = uint32_t(prompt.size()), id = uint32_t(next_id++);
        w.bind(q, n, reused);
        for (uint32_t at = reused; at + 1 < n; at += GRID) w.quantum(q, id, n, at);
        Session s;
        s.prompt = prompt; s.first = int(n) - 1; s.chunk = 8; s.n_gen = n_gen; s.dflash = w.ok(q);
        b.admit({q}, {s});
        while (b.round({q})) {}
        b.release(q);
    }
};
static bool exact(const std::vector<int>& got, int n)
{
    bool ok = !got.empty();
    for (size_t i = 0; i < got.size(); ++i) ok &= got[i] == target(n - 1 + (int)i);
    return ok;
}
static std::vector<int> next_turn(const std::vector<int>& prompt, const std::vector<int>& reply, int extra, int salt)
{
    std::vector<int> p = prompt;
    p.insert(p.end(), reply.begin(), reply.end());
    for (int i = 0; i < extra; ++i) p.push_back((i * 7 + salt) % 977);
    return p;
}

// the bench's "prompt sent again" (#340 done-when): an identical prompt of n tokens hits its own checkpoint at the last
// grid point p <= n - 1 - GRID + ... so its prefill from p taps the whole window when n - 1 - p >= WIN: the drafter
// starts as on a cold prompt, with or without GR8_DFL_REUSE
static void resend(int n)
{
    std::vector<int> p(size_t(n));
    for (int i = 0; i < n; ++i) p[size_t(i)] = (i * 13 + 7) % 991;
    const uint32_t c = g6s2::aligned_point(uint32_t(n), GRID);
    CHECK(c > 0 && int(c) <= n - 1 - WIN);
    for (int reuse : {0, 1})
    {
        Serve f(reuse);
        Batch b(2, 32, f.hooks());
        b.dflash(7, g6rt::SpecLen());
        DflReuse r; r.mode = reuse;
        b.dfl_reuse(r);
        f.turn(b, 0, p, 0, 60);
        CHECK(exact(f.got, n) && f.fill_from == n - 1 - WIN && f.fill_to == n - 1);
        const std::vector<char> cold = f.rounds;
        const std::vector<int> got = f.got;
        f.turn(b, 0, p, c, 60);
        CHECK(exact(f.got, n) && f.got == got);
        CHECK(f.fill_from == n - 1 - WIN && f.fill_to == n - 1);
        CHECK(!f.reqs.empty() && f.reqs[0].lo == n - 1 - WIN);
        CHECK(f.rounds == cold && std::count(f.rounds.begin(), f.rounds.end(), 'd') == (long)f.rounds.size());
    }
}

// an agent's turns whose checkpoint hits land inside the drafter's window: turn 1 ends 101 rows past a grid point, so
// turn 2 (turn 1, its reply and 300 new tokens) hits turn 1's checkpoint there, fewer than WIN rows before its end
static void turns(int grid_points)
{
    const int n1 = grid_points * GRID + 101;
    std::vector<int> p1(size_t(n1));
    for (int i = 0; i < n1; ++i) p1[size_t(i)] = (i * 13 + 7) % 991;
    const uint32_t c1 = g6s2::aligned_point(uint32_t(n1), GRID);
    CHECK(int(c1) == grid_points * GRID);
    for (int reuse : {0, 1})
    {
        Serve f(reuse);
        Batch b(2, 32, f.hooks());
        b.dflash(7, g6rt::SpecLen());
        DflReuse r; r.mode = reuse;
        b.dfl_reuse(r);
        // turn 1: cold, the whole window from the prefill's taps; DFlash2 throughout
        f.turn(b, 0, p1, 0, 60);
        CHECK(exact(f.got, n1));
        CHECK(f.fill_from == n1 - 1 - WIN && f.fill_to == n1 - 1);
        CHECK(!f.reqs.empty() && f.reqs[0].lo == n1 - 1 - WIN);
        CHECK(std::count(f.rounds.begin(), f.rounds.end(), 'd') == (long)f.rounds.size());
        // turn 2 hits turn 1's checkpoint at c1, inside its window
        const std::vector<int> p2 = next_turn(p1, b.result(0), 300, 3);
        const int n2 = int(p2.size()), w2 = n2 - 1 - WIN;
        CHECK(int(c1) > w2);
        f.turn(b, 0, p2, c1, 40);
        CHECK(exact(f.got, n2));
        if (!reuse)
        {   // GR8_DFL_REUSE=0: no drafter for the whole request
            CHECK(f.reqs.empty() && f.fill_from == -1);
            CHECK(std::count(f.rounds.begin(), f.rounds.end(), 'm') == (long)f.rounds.size());
            continue;
        }
        // the reuse: the fill covers [c1, n2 - 1) only, and the drafter's ring rows of turn 1 (the same tokens up to
        // c1) give it the rest of the window: DFlash2 from the first round, no warm-up
        CHECK(f.fill_from == int(c1) && f.fill_to == n2 - 1);
        CHECK(!f.reqs.empty() && f.reqs[0].lo == w2);
        CHECK(b.stats(0).dfl_from == int(c1) && b.stats(0).dfl_lo == w2 && b.stats(0).dfl_warm == 0);
        CHECK(b.stats(0).dfl_rounds == b.stats(0).rounds && b.stats(0).mtp_rounds == 0);
        // turn 3 replays turn 2's first c1 + 1 tokens: a checkpoint holds every prompt row but the last, so no
        // prefill and no fill; the ring (turn 2's) holds the window below c1: DFlash2 from the first round
        std::vector<int> p3(p2.begin(), p2.begin() + c1 + 1);
        f.turn(b, 0, p3, c1, 60);
        CHECK(exact(f.got, int(p3.size())) && f.fill_from == -1);
        CHECK(b.stats(0).dfl_lo == w2 && b.stats(0).dfl_warm == 0 && b.stats(0).dfl_rounds == b.stats(0).rounds);
        // turn 4: another conversation on slot 1 (its tokens differ from 100 on, so the ring is not its context), a
        // hit at c1 of a c1 + 108-token prompt: 107 rows < GR8_DFL_REUSE_MIN, MTP rounds that feed the drafter, then
        // DFlash2 once it holds that many
        std::vector<int> p4(size_t(c1) + 108);
        for (size_t i = 0; i < p4.size(); ++i) p4[i] = i < 100 ? p1[i] : int(i * 11 + 5) % 983;
        f.turn(b, 1, p4, c1, 400);
        CHECK(exact(f.got, int(p4.size())));
        CHECK(f.fill_from == int(c1) && f.fill_to == int(c1) + 107);
        CHECK(b.stats(1).dfl_lo == int(c1) && b.stats(1).dfl_warm == int(c1) + r.min);
        CHECK(!f.rounds.empty() && f.rounds.front() == 'm' && f.rounds.back() == 'd' && b.stats(1).mtp_rounds > 0);
    }
}

int main()
{
    rules();
    resend(32768);      // the bench's 32K and 128K prompts sent again
    resend(131072);
    turns(4);           // 32K and 128K conversations
    turns(16);
    if (fails) { std::printf("flash_dfl_window_test: %d failures\n", fails); return 1; }
    std::printf("flash_dfl_window_test: OK\n");
    return 0;
}
