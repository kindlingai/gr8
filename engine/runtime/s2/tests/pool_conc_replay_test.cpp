// PF-C6-CONC replay: the S2 pool under 4 concurrent sessions (pfc6.py S2 rounds, pool of 1920 physical pages, the request
// sizes of the PFE c6 log). Host only. Without the LRU placement (pool.h lru) the 8192 plusN round loses its probe and two
// fillers: a miss packed over fresh retained runs. Now every resubmit / plusN-resubmit request of the 8192 phase must hit.
// PF_REPLAY_VERBOSE=1 prints one line per request in the g6m4 "# s2 reuse request" format.
#include "s2/pool.h"
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace g6s2;

static std::vector<int> seq(int n, int salt) { std::vector<int> v(n); for (int i = 0; i < n; ++i) v[i] = int(((long long)i * 7919 + salt * 104729 + (i >> 5) * salt) % 100003); return v; }
static std::vector<int> cat(std::vector<int> a, const std::vector<int>& b) { a.insert(a.end(), b.begin(), b.end()); return a; }

struct Live { Run run; std::vector<int> prompt, gen; int skip; };
struct Sim {
    RunPool pool; int grid; uint64_t id = 0; uint32_t guard = 14, maxtok = 16;
    std::vector<Live> live; int log = 1;
    Sim(int g, bool keep) : pool(6144, 1920, 32768), grid(g) { g6rt::PrefixReuse r; r.grid = g; r.noend = true; if (keep) pool.keep(r); }
    uint64_t rows(const std::vector<int>& p) const { return p.size() + maxtok + guard; }
    int lost = 0;                       // requests that should have continued a retained run and did not
    void admit(const std::vector<int>& p, const char* tag, bool expect_hit = false) {
        const uint64_t r = rows(p);
        PoolHit h = pool.match(p, r, int(p.size()) - 1);
        Run run; int skip = 0;
        if (h.skip > 0) { run = pool.take(h, r); skip = h.skip; } else run = pool.reserve(r);
        if (expect_hit && h.skip <= 0) ++lost;
        if (log) std::printf("req %llu %-5s %-14s retained %5d lcp %5d prompt %5zu base %4u pages %3u kept %u\n", (unsigned long long)id, h.why, tag, h.skip, h.lcp, p.size(), run.base_page, run.pages, pool.kept_runs());
        ++id;
        live.push_back(Live{run, p, seq(maxtok, int(id) + 5000), skip});
    }
    void retire_all(bool reverse) {
        std::vector<Live> l = std::move(live); live.clear();
        if (reverse) std::reverse(l.begin(), l.end());
        for (auto& s : l) {
            g6rt::PrefixReuse st; const int n = int(s.prompt.size());
            if (s.skip) st.resume(s.skip);
            if (n - 1 > s.skip) st.prefill_rows(s.skip, n - 1 - s.skip, true);
            st.main_rows(n - 1, 1 + int(s.gen.size())); st.draft_rows(n - 1, 1 + int(s.gen.size()));
            st.finish(s.prompt, s.gen, int(s.gen.size()), true);
            pool.retain(s.run, st);
        }
    }
};

int main(int argc, char** argv) {
    int grid = argc > 1 ? atoi(argv[1]) : 4096;
    Sim S(grid, true); S.guard = argc > 2 ? atoi(argv[2]) : 14;
    auto P = [&](int n, int k) { return seq(n, 7 + k); };
    // requests 0..32 of the real log: sequential, same kind = same prompt (hit), prompt sizes as logged
    struct E { int n, k; };
    std::vector<E> pre = {{44,0},{44,0},{31,1},{31,2},{3521,3},{24665,4},{31,1},{31,2},{3521,3},{24665,4},{31,1},{31,2},{3521,3},{24665,4},{31,1},{31,2},{3521,3},{24665,4},
        {13,5},{8177,6},{8149,7},{43,8},{35,9},{13,5},{32504,10},{32547,11},{45,12},{36,13},{13,5},{129784,14},{129735,15},{45,12},{36,13}};
    S.log = 0;
    for (auto& e : pre) { S.retire_all(false); S.admit(P(e.n, e.k), "pre"); }
    S.log = getenv("PF_REPLAY_VERBOSE") != nullptr;
    auto seqrun = [&](const std::vector<int>& p, const char* tag) { S.retire_all(false); S.admit(p, tag); };
    // S1
    for (int L : {4096, 8192}) {
        const auto b = P(L, 100 + L), e = cat(b, seq(512, 3));
        seqrun(b, "S1 cold"); seqrun(b, "S1 resub"); seqrun(e, "S1 plusN"); seqrun(e, "S1 plusN-resub");
    }
    const std::vector<int> t_cold{90001}, t_re{90002, 90003, 90004}, t_pn{90005, 90006}, t_pnr{90005, 90006, 90007, 90008, 90009};
    int probe_pos[8] = {1, 3, 0, 1, 0, 2, 1, 2}; int ri = 0;
    for (int L : {4096, 8192}) {
        const int plusN = L == 4096 ? 4608 : 8703;
        const auto base = seq(L, 11 + L), ext = cat(base, seq(plusN - L, 12));
        std::vector<std::vector<int>> others; for (int i = 0; i < 3; ++i) others.push_back(seq(L, 100 + i * 10 + L / 4096));
        struct R { const char* name; const std::vector<int>* probe; const std::vector<int>* tail; };
        R rounds[4] = {{"cold", &base, &t_cold}, {"resubmit", &base, &t_re}, {"plusN", &ext, &t_pn}, {"plusN-resub", &ext, &t_pnr}};
        for (auto& r : rounds) {
            if (S.log) std::printf("--- S2 L=%d %s (probe slot %d)\n", L, r.name, probe_pos[ri]);
            std::vector<std::vector<int>> reqs; for (int i = 0; i < 3; ++i) reqs.push_back(cat(others[i], *r.tail));
            reqs.insert(reqs.begin() + probe_pos[ri], *r.probe);
            S.retire_all(false);
            for (size_t k = 0; k < reqs.size(); ++k) S.admit(reqs[k], int(k) == probe_pos[ri] ? "PROBE" : "filler", L == 8192 && std::string(r.name) != "cold");
            ++ri;
        }
    }
    std::printf("pool_conc_replay_test: %s (%d lost)\n", S.lost ? "FAIL" : "PASS", S.lost);
    return S.lost ? 1 : 0;
}
