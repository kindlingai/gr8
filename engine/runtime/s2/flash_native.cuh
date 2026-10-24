// Flash's single resident request uses the existing S2 host service, but not the
// full-GLM paged/multi-session bindings. KDA state is not a paged KV cache.
#pragma once
#if G6_MODEL_FLASH
#include "service.h"
#include "flash_ring.h"
#include "../flash_prefill.cuh"
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
};
inline void serve_flash(FlashContext c, const BootstrapConfig& bc, int chunk) {
    using namespace g6rt;
    const Config& sc = bc.scheduler;
    require(sc.limit() == 1 && sc.prefill_rows > 0, "Flash serve needs one session and prefill");
    const unsigned pages = sc.pool_rows / PAGE;
    RunPool pool(pages, pages, sc.pool_rows);
    FlashRing ring;
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
        require(g.key.slot == 0 && !g.reuse && g.run.first_page == 0,
                "Flash serve requires a fresh identity run");
        const auto list = pool.list(g.run);
        for (size_t i = 0; i < list.size(); ++i) require(list[i] == int(i), "Flash run is not contiguous");
        reset(c.main, c.stream); reset(c.draft.d, c.stream);
        ring.reset(); lp_rows = 0; fence();
    };
    h.retire = [&](const Slot&) { fence(); };
    h.fence = fence;
    h.dispatch.bucket = [&](bool d, int ctx) { return bucket_of(d ? c.draft.d : c.main, ctx); };
    h.dispatch.mode = [&](int, int next) { require(next == -1 || next == 0, "Flash non-solo mode"); fence(); };
    h.dispatch.launch = [&](const Launch& l) {
        require(l.solo == 0 && l.step.sess.size() == 1, "Flash launch is not solo");
        const bool d = l.step.kind != s2::Step::MAIN, chain = l.step.kind == s2::Step::CHAIN;
        StepIn si{}; S2In unused{};
        native_inputs(l, si, unused);
        if (d) {
            si.step = ++dseq;
            *c.draft_input = si;
            std::vector<uint32_t> sizes(3, uint32_t(si.M * HID * 2));
            sizes[2] = uint32_t(glue::head_bytes(si.M, c.draft.d.nc));
            c.draft_fabric.enqueue_step(si.step, sizes);
            c.draft_graph(l.buckets[0], chain, si.M);
        } else {
            si.kda_src = ring.source(si.pos[0]);
            const bool verify = l.event != s2::Step::TAIL;
            require(!verify || si.M <= c.main.kda_ring, "Flash verify exceeds KDA ring");
            for (int r = 0; r < si.M; ++r) si.kslot[r] = verify ? r : 0;
            ring.record(si.pos[0], si.M, verify);
            *c.input = si;
            std::vector<uint32_t> sizes(c.main.ncoll(), uint32_t(si.M * HID * 2));
            sizes.back() = head_size(si, c.main.nc);
            c.main_fabric.enqueue_step(si.step, sizes);
            c.main_graph(l.buckets[0], si.M);
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
        pf_inputs_flash(c.prefill, ma, p.prompt.data() + a, pos.data());
        pf_enqueue_flash(c.prefill_fabric, pfseq, c.main.nl, ma);
        if (mb) {
            pf_inputs_flash(c.prefill2, mb, p.prompt.data() + a + ma, pos.data() + ma);
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
        fprintf(stderr, "# flash serve request %llu prefill %u rows at %u (main + draft): %.3f ms\n",
                (unsigned long long)slot.req.id, rows, first,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    };
    ControlLinks links = bootstrap(bc);
    try {
        wcheck::seal(c.main.rank);
        int dev = 0; G6RT_CK(cudaGetDevice(&dev));
        wcheck::start_watcher(bc.directory, c.main.rank, dev);
        capture_barrier(bc, links);
        Service service(bc, links, std::move(h), chunk);
        service.run();
        links.shutdown();
    } catch (...) { links.shutdown(); throw; }
}
}
#endif
