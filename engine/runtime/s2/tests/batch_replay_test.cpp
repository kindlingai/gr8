// The planner's round log (GR8_S2_REPLAY_LOG, s2_log.h) and its replay (s2/round_replay.h): a log written by a Batch
// replays to the same steps; a changed answer in the log is caught; GR8_S2_FRESH_SPEC makes a request plan as it would
// alone. Host only, a fake engine. With a path argument it also writes the first scenario's log there, for
// round_replay builds to read (scripts/host_tests.sh compares two floating-point contraction modes on it).
#include "s2/round_replay.h"
#include <cstdio>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1
// how many of the drafter's drafts after position q0 are right: varies with the position, so the adaptive choices move
static int right_at(int q0, int bias) { return std::min(7, std::max(0, (q0 * 7 + 3) % 9 - 1 + bias)); }

struct Run
{
    std::vector<std::string> steps;   // each MAIN step's shape and first position
    std::vector<int> got;
};

// requests: each a session run to its end in slot 0, one after another. bias[i]: request i's drafter accuracy shift.
// rec: when set, the Batch logs to it
static Run run(const std::vector<int>& bias, bool fresh, std::FILE* rec, bool dsel = true, int sessions = 1)
{
    Run o;
    int cur = 0;
    Hooks h;
    h.run = [&](const Step& s) {
        std::vector<int> out;
        for (const Row& r : s.r) out.push_back(target(s.kind == Step::MAIN ? r.pos : r.spos - 1));
        if (s.kind == Step::MAIN) o.steps.push_back("m" + std::to_string(s.M()) + "@" + std::to_string(s.r[0].pos));
        return out;
    };
    h.dfl_start = [&](int) { return 0; };
    h.dfl = [&](const DflReq& r) {
        DflDrafts d{};
        const int q0 = r.pos0 + r.rows, ok = right_at(q0, bias[cur]);
        for (int i = 1; i < DFL_B; ++i) { d.tok[i] = i <= ok ? target(q0 + i - 1) : 7; d.lp[i] = i <= ok ? -0.02f * i : -2.5f; }
        return d;
    };
    h.lp = [&](int r) { return -0.05f * (float)(1 + r % 3); };
    h.emit = [&](int, const int* t, int k, int) { o.got.insert(o.got.end(), t, t + k); return false; };
    Batch b(3, 32, h);
    if (rec) b.record(rec);
    b.fresh_spec(fresh);
    g6rt::SpecLen sl;
    sl.on = true;
    sl.set_levels({1, 2, 3, 4, 5, 6, 7});
    b.dflash(7, sl);
    if (dsel)
    {
        DSel d;
        d.on = true;
        d.d.draft_round = 8.1; d.m.draft_round = 5.5; d.m.draft_each = 2.1;
        b.dsel(d);
    }
    KStop ks;
    ks.mode = 1;
    ks.kmax = 3;
    b.kstop(ks);
    for (cur = 0; cur < (int)bias.size(); ++cur)
    {
        std::vector<int> qs;
        std::vector<Session> ss;
        for (int q = 0; q < sessions; ++q)
        {
            Session s;
            for (int i = 0; i < 40; ++i) s.prompt.push_back(target(i - 1 + 100 * q));
            s.first = 39; s.chunk = 8; s.n_gen = 160; s.dflash = sessions == 1;
            qs.push_back(q);
            ss.push_back(s);
        }
        b.admit(qs, ss);
        std::vector<int> order(qs);
        while (b.round(order)) {}
        for (int q : qs) b.release(q);
    }
    return o;
}

static std::vector<std::string> logged(const std::vector<int>& bias, bool fresh, bool dsel = true, int sessions = 1)
{
    std::FILE* f = std::tmpfile();
    run(bias, fresh, f, dsel, sessions);
    std::rewind(f);
    std::vector<std::string> lines = read_lines(f);
    std::fclose(f);
    return lines;
}

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        std::FILE* f = std::fopen(argv[1], "w");
        if (!f) { std::printf("batch_replay_test: cannot write %s\n", argv[1]); return 2; }
        for (const std::string& l : logged({0, -3}, false)) std::fprintf(f, "%s\n", l.c_str());
        std::fclose(f);
    }
    // a solo DFlash2 request with the drafter choice, the confidence stop and adaptive widths, then a second one: the
    // replay asks for every step the log has
    {
        const std::vector<std::string> l = logged({0, -3}, false);
        const ReplayResult r = replay(l);
        CHECK(r.ok);
        CHECK(r.steps > 40 && r.calls > 10);
        if (!r.ok) std::printf("  %s\n", r.error.c_str());
        // a log is deterministic: writing it again gives the same lines
        CHECK(logged({0, -3}, false) == l);
        // one answer changed: the run that follows asks for other steps, and the replay names the line
        std::vector<std::string> bad = l;
        size_t k = 0;
        for (size_t i = 0; i < bad.size() && !k; ++i)
            if (bad[i].rfind("run ", 0) == 0 && i > bad.size() / 3) k = i;
        CHECK(k > 0);
        bad[k] = bad[k].substr(0, bad[k].rfind(' ')) + " 123456";   // the last answer token of that step
        const ReplayResult rb = replay(bad);
        CHECK(!rb.ok && rb.line > k);
        // a missing record: the last call keeps its line but loses its state and learn records
        std::vector<std::string> cut(l.begin(), l.end() - 2);
        CHECK(!replay(cut).ok);
        // learned state that differs (the estimates, the drafter choice, ...) although the steps and the driver's state
        // match: caught at that learn record
        std::vector<std::string> lr = l;
        size_t l0 = 0;
        for (size_t i = lr.size() / 2; i < lr.size() && !l0; ++i) if (lr[i].rfind("learn ", 0) == 0) l0 = i;
        CHECK(l0 > 0);
        lr[l0] = "learn 12345";
        const ReplayResult rl = replay(lr);
        CHECK(!rl.ok && rl.line == l0 + 1);
        // a state the driver would read back differently (one session's length): caught although every step matches
        std::vector<std::string> st = l;
        size_t s0 = 0;
        for (size_t i = st.size() / 2; i < st.size() && !s0; ++i) if (st[i].rfind("state ", 0) == 0) s0 = i;
        CHECK(s0 > 0);
        {
            s2log::In in(st[s0].c_str());
            in.word();
            std::vector<unsigned long long> v;
            in(v);
            v[3] += 1;                    // slot 0's length
            st[s0] = s2log::line("state", v);
        }
        const ReplayResult rs = replay(st);
        CHECK(!rs.ok && rs.line == s0 + 1 && rs.error.find("state") != std::string::npos);
    }
    // record() twice: the second moves the log, the hooks are wrapped once
    {
        std::FILE* f1 = std::tmpfile();
        std::FILE* f2 = std::tmpfile();
        Hooks h;
        h.run = [](const Step& s) { return std::vector<int>(s.M(), 3); };
        h.emit = [](int, const int*, int, int) { return false; };
        Batch b(3, 32, h);
        b.record(f1);
        b.record(f2);
        Session ss;
        ss.prompt.assign(12, 3); ss.first = 11; ss.chunk = 4; ss.n_gen = 6;
        b.admit({0}, {ss});
        while (b.round({0})) {}
        std::rewind(f2);
        const std::vector<std::string> l2 = read_lines(f2);
        const ReplayResult r = replay(l2);
        CHECK(r.ok && r.steps > 0);
        if (!r.ok) std::printf("  %s\n", r.error.c_str());
        std::fclose(f1);
        std::fclose(f2);
    }
    // three sessions in shared MTP rounds (round_all, no drafter)
    {
        const ReplayResult r = replay(logged({0}, false, false, 3));
        CHECK(r.ok && r.steps > 20);
        if (!r.ok) std::printf("  %s\n", r.error.c_str());
    }
    // GR8_S2_FRESH_SPEC: a request after another plans as it would alone; without it the first request's learned widths
    // carry over
    {
        const Run alone = run({-3}, true, nullptr), after = run({2, -3}, true, nullptr);
        const Run alone0 = run({-3}, false, nullptr), after0 = run({2, -3}, false, nullptr);
        const size_t n1 = run({2}, true, nullptr).steps.size(), n10 = run({2}, false, nullptr).steps.size();
        CHECK(std::vector<std::string>(after.steps.begin() + n1, after.steps.end()) == alone.steps);
        CHECK(std::vector<std::string>(after0.steps.begin() + n10, after0.steps.end()) != alone0.steps);
        CHECK(replay(logged({2, -3}, true)).ok);
    }
    std::printf("batch_replay_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
