// Flash on the S2 host service. One session (the default) runs today's solo path on the layers' own tables. Several
// (--s2-sessions 2..8) share the --cap pool through the S2 page tables (s2.cuh, s2_init on the main and the draft
// model): steps with one live session take the solo graphs on that session's pages, steps shared by sessions the
// batched graphs captured here, whose attention reads each row's own table (S2In, the batch views). KDA state is not
// a paged cache: each session has its own kda_ring slots (FlashSlots), reset when it starts.
// GR8_GLM53F_CACHE=aligned (FlashContext::ckpt, flash_checkpoint.h): the pool's prefix-reuse hooks serve Flash
// checkpoints. A prompt's prefill keeps one at its last cold-grid point (its run's pages, its session's first KDA slot
// and raw index rings); a later prompt that starts with its tokens is granted the skip, restored at bind into its own
// run and session slot and prefilled from there.
#pragma once
#if G6_MODEL_FLASH
#include "service.h"
#include "../../knobs-rs/gr8_knobs.h"
#include "flash_ring.h"
#include "flash_checkpoint.cuh"
#include "../flash_prefill.cuh"
#include <map>
#include <atomic>
#include <mutex>
namespace g6s2 {
// The engine's additions to serve_flash's module (g6m4.cu: DFlash2, GR8_GLM53F_KV_CHUNKS borrow-idle); each optional
struct FlashExtras {
    std::function<void(const Grant&)> bind;   // after serve_flash's
    std::function<void(const Slot&)> prefill;  // before serve_flash's prefill quantum
    // a Resize on the engine's tier (FLASH_KV_TIER): true when it was, false for serve_flash's
    std::function<bool(const Resize&)> resize;
    std::function<bool(int)> dfl_ok;
    std::function<int(int)> dfl_start;
    std::function<s2::DflDrafts(const s2::DflReq&, const Sampling&)> dfl;
    std::function<uint32_t()> flash_room;
};
struct FlashContext {
    g6rt::Model& main;
    g6rt::Draft& draft;
    g6rt::FlashPrefill &prefill, &prefill2;
    g6::Fabric &main_fabric, &draft_fabric, &prefill_fabric, &prefill_fabric2;
    cudaStream_t stream;
    g6rt::StepIn *input, *draft_input;
    g6rt::StepOut *output, *draft_output;
    std::function<void(int, int)> main_graph;
    std::function<void(int, bool, int)> draft_graph;
    int split = 0;    // GR8_DEC_SPLIT: the batched main graphs in two parts, as the solo ones
    FlashCheckpoints* ckpt = nullptr;   // GR8_GLM53F_CACHE on: the checkpoint store, else none
    bool terminal = false;              // GR8_GLM53F_CACHE=terminal: also a checkpoint at the end of every request
    // optional: the engine's settings and additions (DFlash2, GR8_GLM53F_KV_CHUNKS borrow-idle), before the service starts
    std::function<void(NativeHooks&, FlashExtras&)> install;
};
inline void serve_flash(FlashContext c, const BootstrapConfig& bc, int chunk) {
    using namespace g6rt;
    const Config& sc = bc.scheduler;
    const bool multi = sc.limit() > 1;
    require(sc.limit() >= 1 && sc.prefill_rows > 0, "Flash serve needs a session and prefill");
    require(!multi || (c.main.s2 && c.draft.d.s2 && c.main.graph && c.draft.d.graph &&
                       sc.limit() <= unsigned(c.main.s2->nslots) && sc.limit() <= unsigned(c.draft.d.s2->nslots) &&
                       c.main.kda_sessions >= int(sc.limit()) && c.main.kda_ring >= 2 &&
                       sc.pool_rows / PAGE >= unsigned(c.main.s2->pool_pages) &&
                       sc.pool_rows / PAGE >= unsigned(c.draft.d.s2->pool_pages)),
            "Flash serve of several sessions needs S2 pools, graphs, --mtp and a KDA ring per session");
    const unsigned pages = sc.pool_rows / PAGE;
    RunPool pool(pages, pages, sc.pool_rows);
    FlashSlots ks(int(sc.limit()), std::max(1, c.main.kda_ring_rest ? c.main.kda_ring_rest : c.main.kda_ring),
                  std::max(1, c.main.kda_ring));
    require(ks.total() <= c.main.kda_st.slots, "Flash serve: the KDA state has fewer slots than the sessions' rings");
    u64 pfseq = 1, pfseq2 = 1, dseq = 0;
    LpStep lp{};
    int lp_rows = 0;
    auto fence = [&] { G6RT_CK(cudaStreamSynchronize(c.stream)); };
    auto check = [&] {
        // Owner OFF leaves these transports unbound; only inspect initialized lanes.
        const auto* owner = c.prefill.pool_transport.fabric;
        const auto* owner2 = c.prefill2.pool_transport.fabric;
        require((!owner || !*owner->dev().error) &&
                (!sc.two_lanes || !owner2 || !*owner2->dev().error), "Flash serve pool-ID fabric error");
        require(!*c.main_fabric.dev().error && !*c.draft_fabric.dev().error &&
                !*c.prefill_fabric.dev().error && (!sc.two_lanes || !*c.prefill_fabric2.dev().error) &&
                !attn_errors(c.main) && !attn_errors(c.draft.d), "Flash serve fabric/attention error");
    };
    // checkpoints: the index is read by the head's admission and written by the prefill, under ck_mu
    FlashCheckpoints* const ck = c.ckpt;
    const uint32_t grid = sc.prefill_rows * (sc.two_lanes ? 2 : 1);
    std::mutex ck_mu;
    std::atomic<uint64_t> claimed{0};   // the entry take() granted, restored by the bind that follows
    // each session's bound run (the retain hook finds the session of a run by it), its checkpoint point (0: none) and
    // cache namespace
    std::vector<Run> bound(size_t(ks.sessions()));
    std::vector<bool> live(size_t(ks.sessions()));
    std::vector<uint32_t> capture_at(size_t(ks.sessions()));
    std::vector<g6rt::CacheNamespace> capture_ns(size_t(ks.sessions()));
    // a quantum that ends at its session's capture_at has run: the capture waits for the next boundary's commits
    // (before_work), so a Resize announced during the prefill and landing at that boundary sizes the store first.
    // Until then no step runs, and no other session's work touches the session's run or KDA slots: rows [0, capture_at)
    // and its first KDA slot are the ones the quantum left
    struct Pending { int q; std::vector<int> tokens; uint64_t req; };
    std::vector<Pending> pending;
    auto take_capture = [&] {
        if (pending.empty()) return;
        std::lock_guard<std::mutex> l(ck_mu);
        for (const Pending& x : pending) {
            const uint32_t at = uint32_t(x.tokens.size() - 1);
            const uint64_t gen = ck->capture(at, ks.base(x.q), x.q, pool.list(bound[size_t(x.q)]), x.tokens, capture_ns[size_t(x.q)], c.stream);
            if (gen) fprintf(stderr, "# flash ckpt request %llu kept %u rows (session %d) as entry %llu (%llu blocks) in %.1f ms; %zu entries\n",
                             (unsigned long long)x.req, at, x.q, (unsigned long long)gen,
                             (unsigned long long)ck->last.blocks, ck->last.ms, ck->index().size());
            else fprintf(stderr, "# flash ckpt request %llu: %u rows do not fit the store (%llu blocks)\n",
                         (unsigned long long)x.req, at, (unsigned long long)ck->blocks());
        }
        pending.clear();
    };
    NativeHooks h;
    // Rows the steps wrote into the attention caches (main and draft) since the last clear: a request clears only the
    // pages that hold them, one page past the last, and every later page is still zero from that clear (the full reset
    // cleared 12 x 436 MB per request at the 262K cap, ~27 ms). -1 = not known (boot): every page.
    // GR8_GLM53F_RESET_ALL=1: every page per request, as before. GR8_GLM53F_RESET_CHECK=1: dies unless the pages past the
    // cleared ones are all zero (a test arm: it reads every cache).
    const bool reset_all = knob::glm53f_reset_all(), reset_check = knob::glm53f_reset_check();
    int dirty = -1;
    unsigned* zflag = nullptr;
    if (reset_check) G6RT_CK(cudaMalloc(&zflag, 4));
    auto wrote = [&](int end) { if (dirty >= 0) dirty = std::max(dirty, end); };
    auto wrote_step = [&](const StepIn& si) {      // a step writes its rows' positions (context = last row's + 1)
        int e = si.context;
        for (int r = 0; r < si.M; ++r) e = std::max(e, si.pos[r] + 1);
        wrote(e);
    };
    h.pool.pool = &pool;   // the driver releases runs on it itself when nothing is retained
    // GR8_GLM53F_CACHE at bind, after the session's cold reset: a hit's entry into the run's rows [0, reuse) and the
    // session's first KDA slot, then the prompt's checkpoint point
    auto bind_ckpt = [&](const Grant& g, int q, const std::vector<int>& list) {
        capture_at[size_t(q)] = 0;
        if (!ck) return;
        capture_ns[size_t(q)] = g6rt::steering::cache_namespace(g.req.steering);
        bool exact = true;   // the rows this request starts from are cold rows (no hit, or an aligned entry)
        if (g.reuse) {
            const uint64_t gen = claimed.exchange(0);
            require(gen != 0, "Flash checkpoint hit without a claimed entry");
            std::lock_guard<std::mutex> l(ck_mu);
            const FlashEntry* x = ck->index().find(gen);
            require(x && x->position == g.reuse, "Flash checkpoint hit differs from its entry");
            exact = !x->terminal;
            ck->restore(gen, ks.base(q), q, list, c.stream);
            wrote((int(g.reuse) + PAGE - 1) / PAGE * PAGE);   // whole pages of both models' caches (one session)
            fprintf(stderr, "# flash ckpt request %llu restored %u rows (session %d, %s entry %llu, %llu blocks) in %.1f ms\n",
                    (unsigned long long)g.req.id, g.reuse, q, exact ? "aligned" : "terminal", (unsigned long long)gen,
                    (unsigned long long)ck->last.blocks, ck->last.ms);
        }
        // at == reuse: the entry that was hit holds these tokens. After a terminal hit no aligned entry is kept: its
        // rows would not be cold rows
        const uint32_t at = aligned_point(g.req.prompt_tokens, grid);
        if (exact && at > g.reuse) capture_at[size_t(q)] = at;
    };
    // the batched graphs (several sessions): kind, M and bucket. Flash's bound layers have no row class, so the main
    // graphs of one M serve every class
    struct Graphs { cudaGraphExec_t a = nullptr, b = nullptr; };
    std::map<std::string, Graphs> graphs;
    auto key = [](char kind, int M, int bucket) { return kind + std::to_string(M) + ":b" + std::to_string(bucket); };
    S2In *layout = nullptr, *draft_layout = nullptr;
    const S2In *old_layout = nullptr, *old_draft_layout = nullptr;
    // A session that verifies more rows than its region holds takes the wide region (FlashSlots::take_wide): DFlash2 for
    // a request admitted alone on a narrow region, or promoted there (GR8_DFL_REUSE=2). A session verifies that many
    // rows only while it is alone (Batch::demote moves it to MTP when a second session arrives), so the previous holder
    // is retiring or free, and before_work took its pending checkpoint at the boundary
    auto widen = [&](const s2::Step& sp, const std::vector<int>& events) {
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const int q = int(sp.sess[j]);
            if (events[j] == s2::Step::TAIL || int(sp.rows[j]) <= ks.width(q)) continue;
            require(sp.sess.size() == 1, "Flash KDA: a verify wider than its session's ring in a shared step");
            if (c.main.kda_st.S) G6RT_CK(kda::copy(c.main.kda_st, 0, ks.base(q), ks.width(q), c.stream));
            const int o = ks.take_wide(q);
            fprintf(stderr, "# flash KDA: session %d takes the wide ring from session %d\n", q, o);
        }
    };
    // several sessions: every batched shape at every context bucket, captured after the bootstrap (its timeout does not
    // cover the capture) and before the weights are sealed, as the full model's capture_all; then serving starts in S2
    // mode with no session (solo -1)
    auto capture = [&](const std::function<void()>& body) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        G6RT_CK(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
        body(); G6RT_CK(cudaStreamEndCapture(c.stream, &graph));
        G6RT_CK(cudaGraphInstantiate(&exec, graph, 0)); G6RT_CK(cudaGraphDestroy(graph));
        return exec;
    };
    auto capture_all = [&] {
        const int nbk = c.main.nbk;
        require(nbk >= 1 && nbk == c.draft.d.nbk && nbk <= NBK, "Flash main/draft context buckets differ");
        const FlashShapes f = flash_shapes(int(sc.limit()), int(sc.mtp), chunk, knob::s2_adapt(), MX);
        // the batched graphs carry no DFlash2 taps (their rows outnumber Dfl::feat's 8); the solo graphs keep them
        auto* const tap = c.main.dfl_tap;
        c.main.dfl_tap = nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        size_t free0 = 0, total = 0, free1 = 0; G6RT_CK(cudaMemGetInfo(&free0, &total));
        c.main.s2on = c.draft.d.s2on = true; fence();
        auto one = [&](bool d, bool chain, int M, int k) {
            Model& model = d ? c.draft.d : c.main;
            S2In h2{};
            h2.ns = chain ? M : 1; h2.cls = s2::row_class(M);
            for (int j = 0; j < h2.ns; ++j) {
                h2.sess[j] = j; h2.row0[j] = chain ? j : 0; h2.rows[j] = chain ? 1 : M; h2.ctx[j] = model.bk[k];
            }
            model.s2->h = h2; model.s2->bucket = k;
            for (int j = 0; j < S2N; ++j) model.s2->bk[j] = k;
            s2_prepare(model, M, c.stream); fence();
            Graphs g;
            if (d) g.a = capture([&] { launch_draft_step(c.draft, M, model.bk[k], chain, c.stream); });
            else if (c.split) {
                g.a = capture([&] { launch_step(c.main, M, model.bk[k], c.stream, 0, c.split); });
                g.b = capture([&] { launch_step(c.main, M, model.bk[k], c.stream, 1, c.split); });
            } else g.a = capture([&] { launch_step(c.main, M, model.bk[k], c.stream, -1, 0); });
            graphs.emplace(key(d ? (chain ? 'c' : 'd') : 'm', M, k), g);
        };
        for (int k = 0; k < nbk; ++k) {
            for (int M : f.mains) one(false, false, M, k);
            for (int M : f.drafts) one(true, false, M, k);
            for (int M : f.chains) one(true, true, M, k);
        }
        fence();
        c.main.dfl_tap = tap;
        G6RT_CK(cudaMemGetInfo(&free1, &total));
        if (c.main.rank == 0)
            fprintf(stderr, "# flash serve capture: %zu shapes x %d buckets (main %zu draft %zu chain %zu; %u sessions, mtp %u, chunk %d) "
                    "in %.2f s, graph memory %.1f MiB\n", graphs.size() / size_t(nbk), nbk, f.mains.size(), f.drafts.size(),
                    f.chains.size(), sc.limit(), sc.mtp, chunk,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    (double(free0) - double(free1)) / (1024.0 * 1024.0));
    };
    auto release = [&] {
        for (auto& item : graphs) {
            if (item.second.a) cudaGraphExecDestroy(item.second.a);
            if (item.second.b) cudaGraphExecDestroy(item.second.b);
        }
        graphs.clear();
        if (layout) { c.main.s2->in = old_layout; c.draft.d.s2->in = old_draft_layout; cudaFreeHost(layout); cudaFreeHost(draft_layout); }
        layout = draft_layout = nullptr;
    };
    if (multi) {
        // the step layouts, host-mapped: the batched graphs' cells read them at launch (as NativeDispatch's)
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&layout), sizeof(S2In), cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&draft_layout), sizeof(S2In), cudaHostAllocMapped));
        *layout = {}; *draft_layout = {};
        old_layout = c.main.s2->in; old_draft_layout = c.draft.d.s2->in;
        c.main.s2->in = layout; c.draft.d.s2->in = draft_layout;
    }
    FlashExtras extra;   // the engine's additions (FlashContext::install)
    const int rank = c.main.rank;
    // Flash's module (ModuleOps): its hooks as members over references to the locals they use. The locals outlive it and the service
#define FLASH_CAPTURES(X) X(c) X(sc) X(multi) X(pool) X(ks) X(pfseq) X(pfseq2) X(dseq) X(lp) X(lp_rows) X(fence) \
    X(check) X(ck) X(ck_mu) X(bound) X(live) X(capture_at) X(pending) X(take_capture) X(reset_all) X(reset_check) \
    X(dirty) X(zflag) X(wrote) X(wrote_step) X(bind_ckpt) X(graphs) X(key) X(layout) X(draft_layout) X(widen) X(rank) \
    X(extra)
    FLASH_CAPTURES(G6S2_CAPTURE_TYPE)
    struct FlashModule : ModuleCtx {
        FLASH_CAPTURES(G6S2_CAPTURE_REF)
        void on_retain(const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) {
            take_capture();
            int q = -1;
            for (int i = 0; i < ks.sessions(); ++i)
                if (live[size_t(i)] && bound[size_t(i)] == r) { q = i; break; }
            require(q >= 0, "Flash checkpoint retain: the run has no bound session");
            live[size_t(q)] = false;
            if (c.terminal && st.history.size() > 1) {
                const uint32_t p = uint32_t(std::min<int>(st.main_end, int(st.history.size()) - 1));
                const FlashRing& ring = ks.rings[size_t(q)];
                int slot = -1;
                try { slot = ks.base(q) + ring.source(int(p)); } catch (const std::exception&) {}
                if (p > 0 && slot >= 0) {
                    const std::vector<int> tokens(st.history.begin(), st.history.begin() + p + 1);
                    std::lock_guard<std::mutex> l(ck_mu);
                    const uint64_t gen = ck->capture(p, slot, q, pool.list(r), tokens, ns, c.stream, true);
                    if (gen) fprintf(stderr, "# flash ckpt terminal kept %u rows (session %d, KDA slot %d) as entry %llu (%llu blocks) in %.1f ms; %zu entries\n",
                                     p, q, slot, (unsigned long long)gen, (unsigned long long)ck->last.blocks, ck->last.ms, ck->index().size());
                    else fprintf(stderr, "# flash ckpt terminal: %u rows do not fit the store (%llu blocks)\n", p, (unsigned long long)ck->blocks());
                } else fprintf(stderr, "# flash ckpt terminal: no state for %u rows (session %d, ring %d+%d)\n", p, q, ring.first, ring.rows);
            }
            pool.release(r);
        }
        void on_bind(const Grant& g) {
            const int q = int(g.key.slot);
            require((!g.reuse || ck) && q < ks.sessions(), "Flash serve requires a fresh run on a session slot");
            const auto list = pool.list(g.run);
            bound[size_t(q)] = g.run; live[size_t(q)] = true;
            if (multi) {
                // the run's pages into slot q's tables, cleared (not reset(), which clears the whole pool and KDA slot 0);
                // the session's first KDA slot from zero (its other slots are written before they are read)
                const int most = std::max(c.main.s2->pool_pages, c.draft.d.s2->pool_pages);
                for (Model* x : {&c.main, &c.draft.d}) {
                    s2_set_pages(*x, q, list);
                    s2_reset_pages(*x, x->s2->run[q], c.stream, most);
                    for (Layer& y : x->layers) if (!y.kda && y.ab.storage) G6RT_CK(g6_attn_bind_raw_reset(&y.ab, q, c.stream));
                }
                if (c.main.kda_st.S) kda::reset(c.main.kda_st, ks.base(q), c.stream);
                ks.reset(q); lp_rows = 0; fence();
                bind_ckpt(g, q, list);
                return;
            }
            require(g.key.slot == 0 && (!g.reuse || ck) && g.run.first_page == 0,
                    "Flash serve requires a fresh identity run");
            for (size_t i = 0; i < list.size(); ++i) require(list[i] == int(i), "Flash run is not contiguous");
            if (dirty < 0 || reset_all) { reset(c.main, c.stream); reset(c.draft.d, c.stream); }
            else {
                const int pg = (dirty + PAGE - 1) / PAGE + 1;
                if (reset_check) {
                    const int lm = reset_dirty_layer(c.main, pg, zflag, c.stream), ld = reset_dirty_layer(c.draft.d, pg, zflag, c.stream);
                    if (lm >= 0 || ld >= 0) {
                        fprintf(stderr, "# GR8_GLM53F_RESET_CHECK: request %llu: nonzero cache past page %d (rows < %d written): main layer %d, draft layer %d\n",
                                (unsigned long long)g.req.id, pg, dirty, lm, ld);
                        require(false, "Flash partial reset: a cache page past the written rows is not zero");
                    }
                }
                reset_pages(c.main, pg, c.stream); reset_pages(c.draft.d, pg, c.stream);
            }
            dirty = 0;
            ks.reset(0); lp_rows = 0; fence();
            bind_ckpt(g, 0, list);
        }
        void on_bind_ext(const Grant& g) { on_bind(g); extra.bind(g); }
        // a session cancelled at the boundary its capture waits for: its rows are still in place (the next bind resets them)
        void on_retire(const Slot&) { fence(); if (ck) take_capture(); }
        void on_before_work() { take_capture(); }
        void on_fence() { fence(); }
        int on_bucket(bool d, int ctx) { return bucket_of(d ? c.draft.d : c.main, ctx); }
        void on_mode(int old, int next) {
            fence();
            if (!multi) { require(next == -1 || next == 0, "Flash non-solo mode"); return; }
            // as the full model's NativeDispatch::mode: the kept draft row moves between the solo path and the session's
            // slot, the solo session's pages go into the layers' own tables (no KV moves)
            if (old >= 0) s2_hprev_from_solo(c.draft, old, c.stream);
            if (next >= 0) {
                const std::vector<int>& run = c.main.s2->run[next];
                require(!run.empty() && run == c.draft.d.s2->run[next], "Flash main/draft survivor runs differ");
                s2_solo_pages(c.main, run); s2_solo_pages(c.draft.d, run);
                s2_hprev_to_solo(c.draft, next, c.stream);
            }
            c.main.s2on = c.draft.d.s2on = next < 0; fence();
        }
        std::vector<int> on_launch(const Launch& l) {
            const bool d = l.step.kind != s2::Step::MAIN, chain = l.step.kind == s2::Step::CHAIN;
            require(multi ? l.solo < 0 || l.step.sess.size() == 1 : l.solo == 0 && l.step.sess.size() == 1,
                    "Flash launch is not solo");
            Model& model = d ? c.draft.d : c.main;
            StepIn si{}; S2In h2{};
            native_inputs(l, si, h2);
            // each row's session: its DSA layers' raw index ring (attn/kvfmt.h)
            for (size_t j = 0; j < l.step.sess.size(); ++j)
                for (int r = l.step.row0[j]; r < l.step.row0[j] + l.step.rows[j]; ++r) si.sess[r] = l.step.sess[j];
            Graphs* g = nullptr;
            if (multi) {
                require(model.s2on == (l.solo < 0), "Flash solo mode differs from host");
                if (l.solo >= 0) si.page_bound = model.s2->pool_pages;   // the solo tables hold -1 past the session's pages
                else {
                    const auto it = graphs.find(key(d ? (chain ? 'c' : 'd') : 'm', si.M, l.bucket));
                    if (it == graphs.end())
                        throw std::runtime_error("Flash launch of an uncaptured shape " + l.graph_key + ": not in the boot capture set");
                    g = &it->second;
                    model.s2->h = h2; model.s2->bucket = l.bucket;
                    for (int j = 0; j < S2N; ++j) model.s2->bk[j] = l.bucket;
                    *(d ? draft_layout : layout) = h2;
                }
            }
            if (d) {
                si.step = ++dseq;
                *c.draft_input = si;
                std::vector<uint32_t> sizes(3, uint32_t(si.M * HID * 2));
                sizes[2] = uint32_t(glue::head_bytes(si.M, c.draft.d.nc));
                c.draft_fabric.enqueue_step(si.step, sizes);
                wrote_step(si);
                if (!g) c.draft_graph(l.buckets[0], chain, si.M);
            } else {
                widen(l.step, l.events);
                ks.step(si, l.step, l.events, s2::Step::TAIL);
                *c.input = si;
                std::vector<uint32_t> sizes(c.main.ncoll(), uint32_t(si.M * HID * 2));
                sizes.back() = head_size(si, c.main.nc);
                c.main_fabric.enqueue_step(si.step, sizes);
                wrote_step(si);
                if (!g) c.main_graph(l.buckets[0], si.M);
            }
            if (g) {
                G6RT_CK(cudaGraphLaunch(g->a, c.stream));
                if (g->b) G6RT_CK(cudaGraphLaunch(g->b, c.stream));
            }
            fence(); check();
            const auto& out = *(d ? c.draft_output : c.output);
            require(!out.attn_err, "Flash step attention error");
            if (!d) {
                g6rt::evprof_read(si.M, c.stream, c.main.rank);
                lp_rows = 0;
                if (std::any_of(l.samples.begin(), l.samples.end(), [](const Sampling& s) { return s.lp >= 0 || s.plp >= 0; })) {
                    lp_fetch(c.main, out, si, lp); lp_rows = si.M;
                }
            }
            return std::vector<int>(out.tok, out.tok + si.M);
        }
        g6rt::LpRow on_logprob(int row, bool target, int k) {
            require(row >= 0 && row < lp_rows, "Flash logprob row outside main step");
            return lp_row(lp, row, target, k);
        }
        void on_prefill(const Slot& slot, const Payload& p, uint32_t first, uint32_t rows) {
            const auto start = std::chrono::steady_clock::now();
            const int a = int(first), ma = std::min(int(rows), int(sc.prefill_rows)), mb = int(rows) - ma;
            require(!mb || (sc.two_lanes && mb <= int(sc.prefill_rows)), "Flash prefill exceeds lanes");
            std::vector<int> pos(rows);
            for (unsigned r = 0; r < rows; ++r) pos[r] = a + int(r);
            wrote(a + int(rows));
            // several sessions: the prefill runs today's path (s2on off) on the session's pages in the layers' own tables
            // (the batched steps read the slots' tables) and its first KDA slot
            const int q = int(slot.key.slot), kslot = multi ? ks.base(q) : 0;
            const bool on = multi && c.main.s2on;
            if (multi) {
                const std::vector<int>& run = c.main.s2->run[q];
                require(!run.empty() && run == c.draft.d.s2->run[q], "Flash main/draft prefill runs differ");
                s2_solo_pages(c.main, run); s2_solo_pages(c.draft.d, run);
                c.main.s2on = c.draft.d.s2on = false;
            }
            pf_inputs_flash(c.prefill, ma, p.prompt.data() + a, pos.data(), kslot, multi ? q : 0);
            pf_enqueue_flash(c.prefill_fabric, c.main, c.prefill, pfseq, c.main.nl, ma);
            if (mb) {
                pf_inputs_flash(c.prefill2, mb, p.prompt.data() + a + ma, pos.data() + ma, kslot, multi ? q : 0);
                pf_enqueue_flash(c.prefill_fabric2, c.main, c.prefill2, pfseq2, c.main.nl, mb);
                pf_step2_flash(c.main, c.prefill, c.prefill2, ma, mb, a + ma, a + ma + mb, pfseq, pfseq2, c.stream);
                pfseq2 += c.main.nl;
            } else pf_step_flash(c.main, c.prefill, ma, a + ma, pfseq, c.stream);
            pfseq += c.main.nl;
            // The cache fill (the default) touches no fabric and no main layer, and leaves its own error word set, so every
            // word check() reads stays set until the next request's reset: one check after the last chunk sees what one per
            // chunk saw, without check()'s blocking copy per layer each MX rows. The fence per chunk stays (k_step_in reads
            // *c.draft_input, which the next chunk rewrites).
            const bool cache = knob::glm53f_draft_kv();
            fence();
            if (!cache) check();
            auto draft = [&](FlashPrefill& fp, int n, int base) {
                if (cache && pf_draft_full() && pf_draft_full_ok(c.draft))   // GR8_GLM53F_PF_DRAFT_FULL (flash_prefill.cuh)
                {
                    const bool chk = pf_draft_full() == 2;
                    DraftFull& F = pf_draft_cache_full(c.draft, fp, n, base, p.prompt.data() + base + 1, chk, c.stream);
                    if (!chk) { dseq += (n + MX - 1) / MX; wrote(base + n); fence(); return; }   // fence: check() reads the error words
                    pf_draft_flash(c.draft, fp, n, base, p.prompt.data() + base + 1, dseq, [&](const StepIn& di) {
                        *c.draft_input = di;
                        wrote_step(di);
                        Model& d = c.draft.d;
                        const bind::Layer& b = d.layers[0].L[di.M];
                        launch_step_in(d, c.stream);
                        launch_draft_embed(c.draft, di.M, false, c.stream);
                        bind::launch(b.kv_a, c.stream);
                        bind::launch(b.idx_k, c.stream);
                        bind::launch(b.idx_g, c.stream);
                        pf_draft_full_cmp(c.draft, F, di.pos[0] - base, di.M, c.stream);
                        fence();
                    }, multi ? q : 0);
                    unsigned dw = 0;
                    G6RT_CK(cudaMemcpy(&dw, F.diff, 4, cudaMemcpyDeviceToHost));
                    fprintf(stderr, "# GR8_GLM53F_PF_DRAFT_FULL check: rows %d at %d, differing kv_a/idx_k/idx_g words %u\n", n, base, dw);
                    return;
                }
                pf_draft_flash(c.draft, fp, n, base, p.prompt.data() + base + 1, dseq, [&](const StepIn& di) {
                    *c.draft_input = di;
                    wrote_step(di);
                    if (cache) pf_draft_cache_flash(c.draft, di.M, di.context, c.stream);
                    else {
                        std::vector<uint32_t> sizes(3, uint32_t(di.M * HID * 2));
                        sizes[2] = uint32_t(glue::head_bytes(di.M, c.draft.d.nc));
                        c.draft_fabric.enqueue_step(di.step, sizes);
                        const int old = c.draft.d.cur;
                        c.draft.d.cur = bucket_of(c.draft.d, di.context);
                        launch_draft_step(c.draft, di.M, di.context, false, c.stream);
                        c.draft.d.cur = old;
                    }
                    fence();
                    if (!cache) check();
                }, multi ? q : 0);
            };
            draft(c.prefill, ma, a);
            if (mb) draft(c.prefill2, mb, a + ma);
            if (cache) check();
            uint32_t& at = capture_at[size_t(q)];
            if (at && first + rows == at) {   // the quantum that ends at the prompt's last grid point
                require(p.prompt.size() > at, "Flash checkpoint past the prompt");
                pending.push_back(Pending{q, std::vector<int>(p.prompt.begin(), p.prompt.begin() + at + 1), slot.req.id});
                at = 0;
            }
            if (on) c.main.s2on = c.draft.d.s2on = true;
            fprintf(stderr, "# flash serve request %llu prefill %u rows at %u (main + draft): %.3f ms\n",
                    (unsigned long long)slot.req.id, rows, first,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        // GR8_PF_PROF=1: each prefill quantum's stage profile
        void on_prefill_prof(const Slot& slot, const Payload& p, uint32_t first, uint32_t rows) {
            on_prefill(slot, p, first, rows);
            const std::string tag = "rank " + std::to_string(c.main.rank) + " request " + std::to_string(slot.req.id) + ",";
            flash_pf_prof().print(tag.c_str());
            flash_pf_prof() = FlashPfProf{};
        }
        void on_prefill_ext(const Slot& slot, const Payload& p, uint32_t first, uint32_t rows) { extra.prefill(slot); on_prefill(slot, p, first, rows); }
        void on_prefill_ext_prof(const Slot& slot, const Payload& p, uint32_t first, uint32_t rows) {
            extra.prefill(slot);
            on_prefill_prof(slot, p, first, rows);
        }
        void on_resize(const Resize& r) {
            if (r.tier != FLASH_CKPT_TIER) throw std::runtime_error("Flash serve: a Resize on tier " + std::to_string(r.tier));
            std::lock_guard<std::mutex> l(ck_mu);
            std::vector<uint64_t> gone;
            const auto t0 = std::chrono::steady_clock::now();
            if (!ck->apply(r, &gone))
                fprintf(stderr, "# rank %d: flash ckpt share: to %llu blocks refused (transaction %llu), the lease keeps its size: %s\n",
                        rank, (unsigned long long)r.slots, (unsigned long long)r.txn, g6kv_tier_error());
            else
                fprintf(stderr, "# rank %d: flash ckpt share: now %llu segments, %llu blocks (%zu entries, %zu dropped) in %.1f ms at %lld ms\n",
                        rank, (unsigned long long)r.segs, (unsigned long long)r.slots, ck->index().size(), gone.size(),
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), (long long)s2_wall_ms());
        }
        void on_resize_kv(const Resize& r) { extra.resize(r); }
        void on_resize_kv_ckpt(const Resize& r) { if (!extra.resize(r)) on_resize(r); }
        bool on_dfl_ok(int q) { return extra.dfl_ok(q); }
        int on_dfl_start(int q) { return extra.dfl_start(q); }
        s2::DflDrafts on_dfl(const s2::DflReq& r, const Sampling& smp) { return extra.dfl(r, smp); }
        uint32_t on_flash_room() { return extra.flash_room(); }
    };
    FlashModule fm{{}, FLASH_CAPTURES(G6S2_CAPTURE_ARG)};
#undef FLASH_CAPTURES
    // the checkpoint store on a group lease (D-331 M8 (c)): it shares its KV node with the full model's pool. Every rank
    // resizes it at the Resize commit; the head reports to the node every GR8_S2_SHARE_MS, offering what it holds only
    // while idle (GR8_KV_SHARE_WHEN=idle, default) or whenever the node asks (always), as the full model's pool does
    if (ck && ck->group()) {
        const bool always = knob::kv_share_when() == "always";
        if (bc.rank == 0)
            h.share = [&, always](bool idle) -> std::optional<Resize> {
                std::pair<uint64_t, std::vector<uint64_t>> a;
                { std::lock_guard<std::mutex> l(ck_mu); a = ck->share_ask(always || idle, !idle); }
                return ck->share_report(FLASH_CKPT_TIER, a.first, a.second);
            };
    }
    if (c.install) c.install(h, extra);
    using M = FlashModule;
    g6s2_module_ops& o = h.module = mod::table(fm);
    if (ck) o.retain = mod::retain<M, &M::on_retain>;
    if (extra.bind) o.bind = mod::bind<M, &M::on_bind_ext>;
    else o.bind = mod::bind<M, &M::on_bind>;
    o.retire = mod::retire<M, &M::on_retire>;
    if (ck) o.before_work = mod::call<M, &M::on_before_work>;
    o.fence = mod::call<M, &M::on_fence>;
    o.bucket = mod::bucket<M, &M::on_bucket>;
    // request grammars (grammar.h): the masks the main head reads (rank.cuh k_grammar_mask), set before every main step
    o.gm_on = c.main.gm_on_host; o.gm_bits = c.main.gm_bits_host; o.gm_words = c.main.gm_words;
    // penalties (penalty.h): Dispatch writes these tables before every main step and rank.cuh k_penalty reads them
    o.pen_rows = c.main.pen_rows_host; o.pen_sess = c.main.pen_sess_host;
    o.mode = mod::mode<M, &M::on_mode>;
    o.launch = mod::launch<M, &M::on_launch>;
    o.logprob = mod::logprob<M, &M::on_logprob>;
    const bool prof = pf_prof().on;
    if (extra.prefill && prof) o.prefill = mod::prefill<M, &M::on_prefill_ext_prof>;
    else if (extra.prefill) o.prefill = mod::prefill<M, &M::on_prefill_ext>;
    else if (prof) o.prefill = mod::prefill<M, &M::on_prefill_prof>;
    else o.prefill = mod::prefill<M, &M::on_prefill>;
    // a Resize on the engine's tier goes to it, any other to the checkpoint store's lease (when it has one)
    const bool ckpt_lease = ck && ck->group();
    if (extra.resize && ckpt_lease) o.resize = mod::resize<M, &M::on_resize_kv_ckpt>;
    else if (extra.resize) o.resize = mod::resize<M, &M::on_resize_kv>;
    else if (ckpt_lease) o.resize = mod::resize<M, &M::on_resize>;
    if (extra.dfl_ok) o.dfl_ok = mod::dfl_ok<M, &M::on_dfl_ok>;
    if (extra.dfl_start) o.dfl_start = mod::dfl_start<M, &M::on_dfl_start>;
    if (extra.dfl) o.dfl = mod::dfl<M, &M::on_dfl>;
    if (extra.flash_room) o.flash_room = mod::flash_room<M, &M::on_flash_room>;
    // Flash's pool policy (PoolOps), as its module above
#define FLASH_POOL_CAPTURES(X) X(pool) X(ck) X(ck_mu) X(claimed)
    G6S2_CAPTURE_TYPE(claimed)
    struct FlashPool : ModuleCtx {
        FLASH_POOL_CAPTURES(G6S2_CAPTURE_REF)
        uint64_t on_digest() {
            if (!ck) return pool.digest();
            std::lock_guard<std::mutex> l(ck_mu);
            return pool.digest() ^ (ck->index().digest() * 0x9E3779B97F4A7C15ull);
        }
        // a retiring run goes back to the pool: its aligned checkpoint, if any, was taken during its prefill. terminal:
        // first keep rows [0, p), p = the rows it consumed (its last output token is pending: token p), with the KDA
        // state its session's ring holds for p (flash_checkpoint.h: not cold-exact). Every rank retires at the same step.
        // A session cancelled at the boundary its aligned capture waits for: that capture first, so the entry for those
        // tokens is the aligned one (a terminal capture of the same tokens then only touches it)
        PoolHit on_match(const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit, const g6rt::CacheNamespace& ns,
                         uint32_t r) {
            std::lock_guard<std::mutex> l(ck_mu);
            const FlashHit f = ck->index().match(prompt, limit, ns);
            PoolHit hit; hit.lcp = f.lcp; hit.why = f.why;
            if (f.hit && pool.fits(rows, worst, r)) { hit.skip = int(f.position); hit.age = f.generation; }
            else if (f.hit) hit.why = "room";
            return hit;
        }
        Run on_take(const PoolHit& hit, uint64_t rows, uint64_t worst, uint32_t r) {
            claimed.store(hit.age);
            return pool.reserve(rows, worst, r);
        }
    };
    FlashPool fp{{}, FLASH_POOL_CAPTURES(G6S2_CAPTURE_ARG)};
#undef FLASH_POOL_CAPTURES
    using P = FlashPool;
    g6s2_pool_ops& po = h.pool.ops = pol::table(fp, pool);
    po.digest = pol::digest<P, &P::on_digest>;
    if (ck) {
        po.match = pol::match<P, &P::on_match>;
        po.take = pol::take<P, &P::on_take>;
    }
    ControlLinks links = bootstrap(bc);
    try {
        if (multi) capture_all();
        // persist=on: settle waits for this rank's load checks, which ran during the capture. agree keeps the entries
        // every rank passed, so every rank has the same index. agree_lines waits with no timeout because ranks check at
        // different speeds. At TP=1 the agreement is with this rank alone
        if (ck && ck->persisted()) {
            const std::string checked = ck->settle();
            const std::string agreed = ck->agree(agree_lines(bc, links, ck->agree_summary(), true));
            fprintf(stderr, "# rank %d: flash ckpt persist: %s; %s\n", c.main.rank, checked.c_str(), agreed.c_str());
        }
        wcheck::seal(c.main.rank);
        int dev = 0; G6RT_CK(cudaGetDevice(&dev));
        wcheck::start_watcher(bc.directory, c.main.rank, dev);
        capture_barrier(bc, links);
        Service service(bc, links, std::move(h), chunk);
        service.run();
        links.shutdown();
    } catch (...) { links.shutdown(); release(); throw; }
    release();
}
}
#endif
