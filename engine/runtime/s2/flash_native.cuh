// Flash on the S2 host service. One session (the default) runs today's solo path on the layers' own tables. Several
// (--s2-sessions 2..8) share the --cap pool through the S2 page tables (s2.cuh, s2_init on the main and the draft
// model): steps with one live session take the solo graphs on that session's pages, steps shared by sessions the
// batched graphs captured here, whose attention reads each row's own table (S2In, the batch views). KDA state is not
// a paged cache: each session has its own kda_ring slots (FlashSlots), reset when it starts.
#pragma once
#if G6_MODEL_FLASH
#include "service.h"
#include "flash_ring.h"
#include "../flash_prefill.cuh"
#include <map>
namespace g6s2 {
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
    int split = 0;    // G6_DEC_SPLIT: the batched main graphs in two parts, as the solo ones
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
    FlashSlots ks(int(sc.limit()), std::max(1, c.main.kda_ring));
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
    NativeHooks h;
    h.pool.free_pages = [&] { return pool.free_pages(); };
    h.pool.room = [&] { return pool.room(); };
    h.pool.reserve = [&](uint64_t n, uint64_t worst) { return pool.reserve(n, worst); };
    h.pool.release = [&](const Run& r) { pool.release(r); };
    h.pool.fits = [&](uint64_t n, uint64_t worst) { return pool.fits(n, worst); };
    h.pool.digest = [&] { return pool.digest(); };
    h.bind = [&](const Grant& g) {
        const int q = int(g.key.slot);
        require(!g.reuse && q < ks.sessions(), "Flash serve requires a fresh run on a session slot");
        const auto list = pool.list(g.run);
        if (!multi) {
            require(q == 0 && g.run.first_page == 0, "Flash serve requires a fresh identity run");
            for (size_t i = 0; i < list.size(); ++i) require(list[i] == int(i), "Flash run is not contiguous");
            reset(c.main, c.stream); reset(c.draft.d, c.stream);
        } else {
            // the run's pages into slot q's tables, cleared (not reset(), which clears the whole pool and KDA slot 0);
            // the session's first KDA slot from zero (its other slots are written before they are read)
            const int most = std::max(c.main.s2->pool_pages, c.draft.d.s2->pool_pages);
            for (Model* x : {&c.main, &c.draft.d}) {
                s2_set_pages(*x, q, list);
                s2_reset_pages(*x, x->s2->run[q], c.stream, most);
            }
            if (c.main.kda_st.S) kda::reset(c.main.kda_st, ks.base(q), c.stream);
        }
        ks.reset(q); lp_rows = 0; fence();
    };
    h.retire = [&](const Slot&) { fence(); };
    h.fence = fence;
    h.dispatch.bucket = [&](bool d, int ctx) { return bucket_of(d ? c.draft.d : c.main, ctx); };
    h.dispatch.mode = [&](int old, int next) {
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
    };
    // the batched graphs (several sessions): kind, M and bucket. Flash's bound layers have no row class, so the main
    // graphs of one M serve every class
    struct Graphs { cudaGraphExec_t a = nullptr, b = nullptr; };
    std::map<std::string, Graphs> graphs;
    auto key = [](char kind, int M, int bucket) { return kind + std::to_string(M) + ":b" + std::to_string(bucket); };
    S2In *layout = nullptr, *draft_layout = nullptr;
    const S2In *old_layout = nullptr, *old_draft_layout = nullptr;
    h.dispatch.launch = [&](const Launch& l) {
        const bool d = l.step.kind != s2::Step::MAIN, chain = l.step.kind == s2::Step::CHAIN;
        require(multi ? l.solo < 0 || l.step.sess.size() == 1 : l.solo == 0 && l.step.sess.size() == 1,
                "Flash launch is not solo");
        Model& model = d ? c.draft.d : c.main;
        StepIn si{}; S2In h2{};
        native_inputs(l, si, h2);
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
            if (!g) c.draft_graph(l.buckets[0], chain, si.M);
        } else {
            ks.step(si, l.step, l.events, s2::Step::TAIL);
            *c.input = si;
            std::vector<uint32_t> sizes(c.main.ncoll(), uint32_t(si.M * HID * 2));
            sizes.back() = head_size(si, c.main.nc);
            c.main_fabric.enqueue_step(si.step, sizes);
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
            lp_rows = 0;
            if (std::any_of(l.samples.begin(), l.samples.end(), [](const Sampling& s) { return s.lp >= 0 || s.plp >= 0; })) {
                lp_fetch(c.main, out, si, lp); lp_rows = si.M;
            }
        }
        return std::vector<int>(out.tok, out.tok + si.M);
    };
    h.dispatch.logprob = [&](int row, bool target, int k) {
        require(row >= 0 && row < lp_rows, "Flash logprob row outside main step");
        return lp_row(lp, row, target, k);
    };
    h.dispatch.prefill = [&](const Slot& slot, const Payload& p, uint32_t first, uint32_t rows) {
        const auto start = std::chrono::steady_clock::now();
        const int a = int(first), ma = std::min(int(rows), int(sc.prefill_rows)), mb = int(rows) - ma;
        require(!mb || (sc.two_lanes && mb <= int(sc.prefill_rows)), "Flash prefill exceeds lanes");
        std::vector<int> pos(rows);
        for (unsigned r = 0; r < rows; ++r) pos[r] = a + int(r);
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
        pf_inputs_flash(c.prefill, ma, p.prompt.data() + a, pos.data(), kslot);
        pf_enqueue_flash(c.prefill_fabric, pfseq, c.main.nl, ma);
        if (mb) {
            pf_inputs_flash(c.prefill2, mb, p.prompt.data() + a + ma, pos.data() + ma, kslot);
            pf_enqueue_flash(c.prefill_fabric2, pfseq2, c.main.nl, mb);
            pf_step2_flash(c.main, c.prefill, c.prefill2, ma, mb, a + ma, a + ma + mb, pfseq, pfseq2, c.stream);
            pfseq2 += c.main.nl;
        } else pf_step_flash(c.main, c.prefill, ma, a + ma, pfseq, c.stream);
        pfseq += c.main.nl;
        // The cache fill (the default) touches no fabric and no main layer, and leaves its own error word set, so every
        // word check() reads stays set until the next request's reset: one check after the last chunk sees what one per
        // chunk saw, without check()'s blocking copy per layer each MX rows. The fence per chunk stays (k_step_in reads
        // *c.draft_input, which the next chunk rewrites).
        const bool cache = !getenv("G6_FLASH_DRAFT_KV") || atoi(getenv("G6_FLASH_DRAFT_KV"));
        fence();
        if (!cache) check();
        auto draft = [&](FlashPrefill& fp, int n, int base) {
            pf_draft_flash(c.draft, fp, n, base, p.prompt.data() + base + 1, dseq, [&](const StepIn& di) {
                *c.draft_input = di;
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
            });
        };
        draft(c.prefill, ma, a);
        if (mb) draft(c.prefill2, mb, a + ma);
        if (cache) check();
        if (on) c.main.s2on = c.draft.d.s2on = true;
        fprintf(stderr, "# flash serve request %llu prefill %u rows at %u (main + draft): %.3f ms\n",
                (unsigned long long)slot.req.id, rows, first,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
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
        const FlashShapes f = flash_shapes(int(sc.limit()), int(sc.mtp), chunk, s2::adapt_env(), MX);
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
    ControlLinks links = bootstrap(bc);
    try {
        if (multi) capture_all();
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
