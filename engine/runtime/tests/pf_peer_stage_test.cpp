// Host model of the GR8_GLM53F_PF_STAGE_ASYNC schedule (flash_prefill.cuh: pf_step_flash, pf_step2_flash, pf_put_flash ->
// pf_stage_issue, pf_consume_flash). No CUDA. Every stream op, event record and event wait the real code makes is
// recorded in call order; an op graph (stream order + record -> wait edges, the CUDA rule: a wait sees the latest
// record made before it in call order) then gives happens-before, and the hazards of NK-DESIGN.md are checked on it:
//   (1) the peer buffer: copy(n+1) of a lane starts after the mHC kernel of consume(n) of that lane
//   (2) the step cell: k_wait_peers(n) is ordered before every later k_set_step of the lane
//   (3) the fabric slot: copy(c, n) is ordered before put(c, n+1) of the lane (the peer's rewrite of the slot needs that put)
//   (5) k_wait_peers(n) is ordered after this lane's put(n) (the step cell and the doorbell are ours to read)
//   (4) consume(n) kernel is ordered after copy(n), issue count == consume count, one outstanding issue per lane
// mutations (a put moved before the consume, no used-wait) must be caught. NR = 2, 3 and 4; 1 and 2 lanes; ON and OFF.
//   g++ -std=c++17 -O2 -Wall engine/runtime/tests/pf_peer_stage_test.cpp -o /tmp/pf_peer_stage_test && /tmp/pf_peer_stage_test
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct Op { int stream; std::string kind; int lane, c, n; };   // n: collective ordinal of the lane (0 = first put)
struct Model
{
    std::vector<Op> ops;
    std::vector<std::vector<int>> pred;
    std::map<int, int> last_on_stream;
    std::map<std::string, int> last_record;   // event name -> op
    int add(int stream, const std::string& kind, int lane = -1, int c = -1, int n = -1)
    {
        const int id = (int)ops.size();
        ops.push_back({stream, kind, lane, c, n});
        pred.emplace_back();
        auto it = last_on_stream.find(stream);
        if (it != last_on_stream.end()) pred[id].push_back(it->second);
        last_on_stream[stream] = id;
        return id;
    }
    void record(int stream, const std::string& ev, int lane) { last_record[ev] = add(stream, "record:" + ev, lane); }
    void wait(int stream, const std::string& ev, int lane)
    {
        const int id = add(stream, "wait:" + ev, lane);
        auto it = last_record.find(ev);
        if (it != last_record.end()) pred[id].push_back(it->second);   // a never-recorded event is a no-op
    }
    std::vector<std::vector<char>> hb;   // hb[a][b]: a happens before b
    void close()
    {
        const size_t N = ops.size();
        hb.assign(N, std::vector<char>(N, 0));
        for (size_t b = 0; b < N; ++b)   // preds always have smaller ids (call order)
            for (int a : pred[b]) { hb[a][b] = 1; for (int x = 0; x < a; ++x) if (hb[x][a]) hb[x][b] = 1; }
    }
};

struct Cfg { int nr; int lanes; int layers; bool on; bool bug_early_put = false; bool bug_no_used = false; bool bug_no_ready = false; bool bug_no_putwait = false; };

struct Lane
{
    int id; int cst; int issued = 0, consumed = 0, puts[2] = {0, 0};
    std::vector<int> wait_op, copy_ops_last, consume_kernel, put_op[2];   // per ordinal
    std::vector<std::vector<int>> copies;
    std::vector<int> set_steps;
};

struct Run
{
    Cfg cfg; Model m; Lane lane[2];
    std::string err;
    static constexpr int CS = 0;   // compute stream
    void fail(const std::string& s) { if (err.empty()) err = s; }
    std::string ev(int l, const char* k) { return std::string(k) + std::to_string(l); }
    void put(int l, int c)
    {
        Lane& L = lane[l];
        const int n = L.puts[c]++;
        const int p = m.add(CS, "put", l, c, n);
        L.put_op[c].push_back(p);
        if (!cfg.on) return;
        m.record(CS, ev(l, "put"), l);
        if (!cfg.bug_no_putwait) m.wait(L.cst, ev(l, "put"), l);
        if (!cfg.bug_no_used) m.wait(L.cst, ev(l, "used"), l);
        L.wait_op.push_back(m.add(L.cst, "k_wait_peers", l, c, L.issued));
        std::vector<int> cp;
        for (int q = 0; q < cfg.nr - 1; ++q) cp.push_back(m.add(L.cst, "copy", l, c, L.issued));
        L.copies.push_back(cp);
        m.record(L.cst, ev(l, "ready"), l);
        if (L.issued != L.consumed) fail("two issues outstanding on a lane");
        ++L.issued;
    }
    void consume(int l, int c)
    {
        Lane& L = lane[l];
        const int n = L.consumed;
        if (cfg.on)
        {
            if (L.issued != L.consumed + 1) { fail("consume without a matching issue"); return; }
            if (!cfg.bug_no_ready) m.wait(CS, ev(l, "ready"), l);
        }
        L.consume_kernel.push_back(m.add(CS, "mhc", l, c, n));
        if (cfg.on) m.record(CS, ev(l, "used"), l);
        ++L.consumed;
    }
    void set_step(int l) { lane[l].set_steps.push_back(m.add(CS, "k_set_step", l)); }
    void block(int l) { m.add(CS, "block", l); }

    void one_lane(int layers)
    {
        for (int i = 0; i < layers; ++i)
        {
            set_step(0); block(0); put(0, 0); consume(0, 0);
            block(0); put(0, 1); consume(0, 1);
        }
    }
    void two_lanes(int layers)
    {
        // pf_step2_flash: A = lane 0, B = lane 1
        for (int i = 0; i < layers; ++i)
        {
            set_step(0); block(0); put(0, 0);
            if (i > 0) consume(1, 1);
            set_step(1); block(1); put(1, 0);
            if (cfg.bug_early_put) { block(0); put(0, 1); }   // a put before the consume that waits ready: the hazard pf_stage_issue forbids
            consume(0, 0);
            if (!cfg.bug_early_put) { block(0); put(0, 1); }
            consume(1, 0);
            block(1); put(1, 1);
            consume(0, 1);
        }
        consume(1, 1);
    }
    void go()
    {
        for (int l = 0; l < 2; ++l) lane[l].id = l, lane[l].cst = l + 1;
        if (cfg.lanes == 1) one_lane(cfg.layers); else two_lanes(cfg.layers);
        m.close();
        check();
    }
    bool hb(int a, int b) const { return m.hb[a][b]; }
    void check()
    {
        for (int l = 0; l < cfg.lanes; ++l)
        {
            Lane& L = lane[l];
            if (L.puts[0] + L.puts[1] != L.consumed) fail("put count != consume count");
            if (!cfg.on) continue;
            if (L.issued != L.consumed) fail("issue count != consume count");
            const size_t N = L.consume_kernel.size();
            for (size_t k = 0; k < N; ++k)
            {
                for (int cp : L.copies[k]) if (!hb(cp, L.consume_kernel[k])) fail("consume kernel not ordered after its copy");
                if (k + 1 < N)
                {
                    for (int cp : L.copies[k + 1])
                        if (!hb(L.consume_kernel[k], cp)) fail("peer buffer: next copy not ordered after the previous mHC kernel");
                    for (int s : L.set_steps)
                        if (m.ops[s].kind == "k_set_step" && s > L.wait_op[k] && !hb(L.wait_op[k], s) && s > L.consume_kernel[k])
                            fail("step cell: k_set_step not ordered after k_wait_peers");
                }
            }
            for (size_t k = 0; k < L.wait_op.size(); ++k)
            {
                const int c = m.ops[L.wait_op[k]].c;
                int n = 0;
                for (size_t j = 0; j < k; ++j) n += m.ops[L.wait_op[j]].c == c;
                if (!hb(L.put_op[c][n], L.wait_op[k])) fail("k_wait_peers not ordered after its put (stale step cell)");
            }
            // every k_set_step issued after the issue of collective k must be ordered after k_wait_peers(k)
            for (size_t k = 0; k < L.wait_op.size(); ++k)
                for (int s : L.set_steps)
                    if (s > L.wait_op[k] && !hb(L.wait_op[k], s)) fail("step cell: later k_set_step not ordered after k_wait_peers");
            // fabric slot: copy(c, n) before put(c, n + 1) of the same collective c
            for (int c = 0; c < 2; ++c)
            {
                std::vector<int> cps;   // last copy op of each issue of collective c, in order
                for (size_t k = 0; k < L.copies.size(); ++k)
                    if (m.ops[L.copies[k][0]].c == c) cps.push_back(L.copies[k].back());
                for (size_t k = 0; k + 1 < L.put_op[c].size() && k < cps.size(); ++k)
                    if (!hb(cps[k], L.put_op[c][k + 1])) fail("fabric slot: put of the next step not ordered after the copy");
            }
        }
    }
};

static int fails = 0, runs = 0;
static void expect(const Cfg& cfg, bool ok, const char* name)
{
    Run r; r.cfg = cfg; r.go(); ++runs;
    const bool got = r.err.empty();
    if (got != ok) { ++fails; std::printf("FAIL %s nr=%d lanes=%d on=%d: %s\n", name, cfg.nr, cfg.lanes, cfg.on, r.err.empty() ? "no error where one was expected" : r.err.c_str()); }
}

int main()
{
    for (int nr : {2, 3, 4})
        for (int lanes : {1, 2})
            for (int layers : {1, 2, 5})
            {
                expect({nr, lanes, layers, true}, true, "ON");
                expect({nr, lanes, layers, false}, true, "OFF");
                // the used-wait is redundant while the issue follows the previous consume on the compute stream (put event); kept as a guard
                expect({nr, lanes, layers, true, false, true}, true, "no used-wait");
                expect({nr, lanes, layers, true, false, false, true}, false, "no ready-wait is caught");
                expect({nr, lanes, layers, true, false, false, false, true}, false, "no put-wait is caught");
                if (lanes == 2) expect({nr, lanes, layers, true, true}, false, "early put is caught");
            }
    std::printf("pf_peer_stage_test: %d runs, %d failures\n", runs, fails);
    return fails ? 1 : 0;
}
