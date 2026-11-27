// ERNIE-5.5-Flash-Preview S2 host service adapter: wraps ernie55rt::Engine with dynamic RunPool page tables
// (BF16 / FP8 / INT4 DSA KV + HCA C128 SWA/compressed KV) for both Spark resident & 5090 hybrid modes.
#ifndef G6_S2_ERNIE55_NATIVE_CUH
#define G6_S2_ERNIE55_NATIVE_CUH
#include "service.h"
#include "../ernie55_step.cuh"
#include <algorithm>
#include <vector>

namespace g6s2 {

struct Ernie55Context {
    ernie55rt::Engine& engine;
    int rank = 0;
};

inline void serve_ernie55(Ernie55Context c, const BootstrapConfig& bc, int chunk) {
    const Config& sc = bc.scheduler;
    const int slots = c.engine.o.slots;
    const int max_rows = c.engine.o.max_rows;
    require(sc.max_sessions == unsigned(slots) && sc.max_rows <= unsigned(max_rows),
            "ernie55: --s2-sessions and --max-rows mismatch");
    const unsigned pool_pages = unsigned(c.engine.pool_pages);
    const unsigned max_pps = unsigned(c.engine.max_pps);
    RunPool pool(pool_pages, pool_pages, pool_pages * geom::KV_PAGE);

    std::vector<ernie55rt::RowOut> out;
    std::vector<int> targets;
    int lp_rows = 0;

    NativeHooks h;
    h.pool.pool = &pool;
    h.pool.grow = true;
    h.pool.max_run_pages = max_pps;
    h.pool.status = true;

    h.bind = [&](const Grant& g) {
        const int q = int(g.key.slot);
        require(q >= 0 && q < slots, "ernie55 bind: bad slot");
        c.engine.reset_slot(q);
        c.engine.set_slot_pages(q, pool.list(g.run));
        ERNIE55_CK(cudaStreamSynchronize(c.engine.stream));
    };

    h.grown = [&](unsigned q, const Run& r) {
        require(int(q) >= 0 && int(q) < slots, "ernie55 grown: bad slot");
        c.engine.set_slot_pages(int(q), pool.list(r));
    };

    h.retire = [&](const Slot&) {
        ERNIE55_CK(cudaDeviceSynchronize());
    };
    h.fence = []() { ERNIE55_CK(cudaDeviceSynchronize()); };
    h.dispatch.bucket = [](bool, int) { return 0; };
    h.dispatch.mode = [](int, int) {};

    h.dispatch.launch = [&](const Launch& l) -> std::vector<int> {
        const s2::Step& sp = l.step;
        const int M = sp.M();
        require(M >= 1 && M <= max_rows && l.samples.size() >= size_t(M), "ernie55 launch: bad M");
        std::vector<ernie55::StepSeg> segs;
        std::vector<int> toks(size_t(M), 0);
        std::vector<ernie55rt::RowSample> sm(static_cast<size_t>(M));
        targets.assign(size_t(M), 0);
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const int q = sp.sess[j], r0 = sp.row0[j], n = sp.rows[j];
            const int pos0 = sp.r[size_t(r0)].pos;
            segs.push_back({q, r0, pos0, n});
            for (int r = r0; r < r0 + n; ++r) {
                const s2::Row& x = sp.r[size_t(r)];
                toks[size_t(r)] = x.tok;
                targets[size_t(r)] = x.tgt;
                const Sampling& s = l.samples[size_t(r)];
                sm[size_t(r)] = ernie55rt::RowSample{s.inv_temp, s.top_p, s.seed, uint32_t(x.spos)};
            }
        }
        c.engine.forward(segs, toks.data(), sm.data(), out, false, nullptr, true);
        lp_rows = M;
        std::vector<int> ids(static_cast<size_t>(M));
        for (int r = 0; r < M; ++r) ids[size_t(r)] = out[size_t(r)].tok;
        return ids;
    };

    h.dispatch.logprob = [&](int row, bool target, int k) -> g6rt::LpRow {
        require(row >= 0 && row < lp_rows, "ernie55 logprob: bad row");
        const ernie55rt::RowOut& o = out[size_t(row)];
        const int id = target ? targets[size_t(row)] : o.tok;
        g6rt::LpRow e;
        e.n = std::max(0, std::min({k, int(g6rt::LP_MAX), int(o.top.size())}));
        for (int i = 0; i < e.n; ++i) { e.id[i] = o.top[size_t(i)].id; e.tlp[i] = o.top[size_t(i)].logit - o.lse; }
        e.lp = o.top.empty() ? 0.f : o.top.back().logit - o.lse;
        for (size_t i = 0; i < o.top.size(); ++i)
            if (o.top[i].id == id) { e.lp = o.top[i].logit - o.lse; e.rank = i < size_t(g6rt::LP_MAX) ? int(i) + 1 : -1; break; }
        return e;
    };

    h.dispatch.prefill = [&](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
        const int q = int(s.key.slot);
        require(size_t(first) + rows <= p.prompt.size() && q >= 0 && q < slots, "ernie55 prefill: out of range");
        uint32_t at = first;
        const uint32_t limit = first + rows;
        while (at < limit) {
            const uint32_t step_lim = std::min<uint32_t>(limit, at + uint32_t(max_rows));
            const int n = int(step_lim - at);
            std::vector<ernie55::StepSeg> segs = {{q, 0, int(at), n}};
            c.engine.forward(segs, &p.prompt[at], nullptr, out, true);
            at = step_lim;
        }
        lp_rows = 0;
    };

    ControlLinks links = bootstrap(bc);
    try {
        capture_barrier(bc, links);
        Service service(bc, links, std::move(h), chunk);
        service.run();
        links.shutdown();
    } catch (...) { links.shutdown(); throw; }
}

} // namespace g6s2
#endif
