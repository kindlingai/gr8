// Tencent Hy4-flash S2 host service adapter: wraps hy4rt::Engine with dynamic RunPool page tables
// (BF16 / FP8 / INT4 KV up to 1M context) and native built-in 0.85B DFly2 speculative drafting!
#ifndef G6_S2_HY4_NATIVE_CUH
#define G6_S2_HY4_NATIVE_CUH
#include "service.h"
#include "../hy4_step.cuh"
#include <algorithm>
#include <array>
#include <vector>

namespace g6s2 {

struct Hy4Context {
    hy4rt::Engine& engine;
    int rank = 0;
};

inline void serve_hy4(Hy4Context c, const BootstrapConfig& bc, int chunk) {
    const Config& sc = bc.scheduler;
    const int slots = c.engine.o.slots;
    const int max_rows = c.engine.o.max_rows;
    require(sc.max_sessions == unsigned(slots) && sc.max_rows <= unsigned(max_rows),
            "hy4: --s2-sessions and --max-rows mismatch");
    const unsigned pool_pages = unsigned(c.engine.pool_pages);
    const unsigned max_pps = unsigned(c.engine.max_pps);
    RunPool pool(pool_pages, pool_pages, pool_pages * geom::KV_PAGE);

    const bool use_dfly = c.engine.has_dfly && c.engine.o.dfly_tokens > 0;
    const int dfly_k = use_dfly ? std::min(c.engine.o.dfly_tokens, int(s2::DFL_B) - 1) : 0;

    struct PendingVerify {
        bool active = false;
        int pos0 = 0;
        int rows = 0;
        std::vector<int> toks;
        std::array<int32_t, 2> saved_eng{};
    };
    std::vector<PendingVerify> pend(static_cast<size_t>(slots));
    std::vector<int> last_snap_off(size_t(slots), 0);

    std::vector<hy4rt::RowOut> out;
    std::vector<int> targets;
    int lp_rows = 0;

    auto settle_pending = [&](int q, int next_pos) {
        auto& pv = pend[size_t(q)];
        if (!pv.active) return;
        const int kept = next_pos - pv.pos0;
        if (kept > 0 && kept < pv.rows) {
            c.engine.dfly_rollback_slot(q, pv.pos0, kept, pv.toks.data(), pv.saved_eng);
        }
        pv.active = false;
    };

    NativeHooks h;
    h.pool.pool = &pool;
    h.pool.grow = true;
    h.pool.max_run_pages = max_pps;
    h.pool.status = true;

    h.bind = [&](const Grant& g) {
        const int q = int(g.key.slot);
        require(q >= 0 && q < slots, "hy4 bind: bad slot");
        pend[size_t(q)] = PendingVerify{};
        last_snap_off[size_t(q)] = 0;
        c.engine.reset_slot(q);
        c.engine.set_slot_pages(q, pool.list(g.run));
        HY4_CK(cudaStreamSynchronize(c.engine.stream));
    };

    h.grown = [&](unsigned q, const Run& r) {
        require(int(q) >= 0 && int(q) < slots, "hy4 grown: bad slot");
        c.engine.set_slot_pages(int(q), pool.list(r));
    };

    h.retire = [&](const Slot& s) {
        if (int(s.key.slot) >= 0 && int(s.key.slot) < slots) pend[size_t(s.key.slot)] = PendingVerify{};
        HY4_CK(cudaDeviceSynchronize());
    };
    h.fence = []() { HY4_CK(cudaDeviceSynchronize()); };
    h.dispatch.bucket = [](bool, int) { return 0; };
    h.dispatch.mode = [](int, int) {};

    if (use_dfly) {
        h.dfl_kv = dfly_k;
        h.dispatch.dfl_ok = [&](int q) { return q >= 0 && q < slots; };
        h.dispatch.dfl_start = [&](int) -> int { return 0; };
        h.dispatch.dfl = [&](const s2::DflReq& r, const Sampling& samp) -> s2::DflDrafts {
            const int q = r.sess;
            require(q >= 0 && q < slots, "hy4 dfl: bad slot");
            int snap_off = last_snap_off[size_t(q)];
            auto& pv = pend[size_t(q)];
            if (pv.active) {
                const int kept = r.rows;
                if (kept > 0 && kept < pv.rows) {
                    c.engine.dfly_rollback_slot(q, pv.pos0, kept, pv.toks.data(), pv.saved_eng);
                }
                if (kept > 0) snap_off = std::min(kept - 1, c.engine.dfly_w_max - 1);
                pv.active = false;
            }
            s2::DflDrafts o{};
            if (!r.query) return o;
            const int anchor = r.pos0 + r.rows;
            const int K = std::min(dfly_k, std::max(0, c.engine.o.cap - anchor - 1));
            o.tok[0] = r.tok0;
            o.lp[0] = 0.0f;
            if (K <= 0) return o;
            hy4rt::RowSample sm{samp.inv_temp, samp.top_p, samp.seed, uint32_t(anchor + 1)};
            std::vector<hy4rt::RowOut> dro;
            const bool want_lp = (samp.inv_temp > 0.f);
            std::vector<int> drafts = c.engine.dfly_propose(
                q, r.tok0, anchor, snap_off, K,
                want_lp ? &sm : nullptr, want_lp ? &dro : nullptr);
            for (int i = 0; i < K && i + 1 < s2::DFL_B; ++i) {
                o.tok[i + 1] = drafts[size_t(i)];
                float lp = 0.0f;
                if (want_lp && i < int(dro.size())) {
                    for (const auto& cd : dro[size_t(i)].top) {
                        if (cd.id == drafts[size_t(i)]) { lp = cd.logit - dro[size_t(i)].lse; break; }
                    }
                }
                o.lp[i + 1] = lp;
            }
            return o;
        };
        if (c.rank == 0) {
            std::fprintf(stderr, "# hy4 s2: built-in 0.85B DFly2 enabled (K=%d, kv=%s, mx8=%d)\n",
                         dfly_k, hy4rt::kv_fmt_name(c.engine.o.kv_fmt), c.engine.o.mx8_dense ? 1 : 0);
        }
    }

    h.dispatch.launch = [&](const Launch& l) -> std::vector<int> {
        const s2::Step& sp = l.step;
        const int M = sp.M();
        require(M >= 1 && M <= max_rows && l.samples.size() >= size_t(M), "hy4 launch: bad M");
        std::vector<hy4::StepSeg> segs;
        std::vector<int> toks(size_t(M), 0);
        std::vector<hy4rt::RowSample> sm(static_cast<size_t>(M));
        targets.assign(size_t(M), 0);
        bool any_verify = false;
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const int q = sp.sess[j], r0 = sp.row0[j], n = sp.rows[j];
            const int pos0 = sp.r[size_t(r0)].pos;
            if (use_dfly) settle_pending(q, pos0);
            segs.push_back({q, r0, pos0, n});
            for (int r = r0; r < r0 + n; ++r) {
                const s2::Row& x = sp.r[size_t(r)];
                toks[size_t(r)] = x.tok;
                targets[size_t(r)] = x.tgt;
                const Sampling& s = l.samples[size_t(r)];
                sm[size_t(r)] = hy4rt::RowSample{s.inv_temp, s.top_p, s.seed, uint32_t(x.spos)};
            }
            if (use_dfly && sp.round >= 0 && n > 1 && n <= c.engine.dfly_w_max) {
                any_verify = true;
                auto& pv = pend[size_t(q)];
                pv.active = true;
                pv.pos0 = pos0;
                pv.rows = n;
                pv.toks.assign(toks.begin() + r0, toks.begin() + r0 + n);
                pv.saved_eng = c.engine.eng_ctx[size_t(q)];
                c.engine.dfly_save_idx_raw(q);
            }
            last_snap_off[size_t(q)] = c.engine.snap_off_for_seg_len(n);
        }
        c.engine.forward(segs, toks.data(), sm.data(), out, false, nullptr, any_verify);
        lp_rows = M;
        std::vector<int> ids(static_cast<size_t>(M));
        for (int r = 0; r < M; ++r) ids[size_t(r)] = out[size_t(r)].tok;
        return ids;
    };

    h.dispatch.logprob = [&](int row, bool target, int k) -> g6rt::LpRow {
        require(row >= 0 && row < lp_rows, "hy4 logprob: bad row");
        const hy4rt::RowOut& o = out[size_t(row)];
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
        require(size_t(first) + rows <= p.prompt.size() && q >= 0 && q < slots, "hy4 prefill: out of range");
        pend[size_t(q)] = PendingVerify{};
        uint32_t at = first;
        const uint32_t limit = first + rows;
        while (at < limit) {
            const uint32_t step_lim = std::min<uint32_t>(limit, at + uint32_t(max_rows));
            const int n = int(step_lim - at);
            std::vector<hy4::StepSeg> segs = {{q, 0, int(at), n}};
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
