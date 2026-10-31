// DeepSeek-V4.1-Flash on the S2 host service, serve v1: ONE session at position 0, no prefix reuse, MTP 0 (or DSpark MTP 1..5 with a draft Model), Engram on or off (the token ring of ds41_engram_ops.cuh), split 0.
// This is serve_flash's solo path (flash_native.cuh) cut to what DS41 has. The prompt runs as MAIN decode steps (every DS41 prompt is
// decode_prompt, so the Driver never calls the prefill hook) unless BootstrapConfig::prefill_min admits it: then its first n - 1 rows go
// to Ds41Context::prefill in chunks of at most Config::prefill_rows (ds41_prefill.h) and its last token is a MAIN row. A step is one captured graph (or, without --graph, one eager step) the caller launches (g6m4.cu),
// and the attention / index caches are B4's / B5's own buffers that reset() clears between requests (rank.cuh). A Step::DRAFT or
// Step::CHAIN never reaches this hook at K = 0 (s2_batch.h); if one did, it launches nothing and returns zeros. With DSpark (Ds41Context::dsp,
// GR8_DS_MTP) they go to the round runner (ds41_dspark_round.cuh).
// S2 prefix snapshots are opt-in; without --reuse-prefix the old bind/reset path is unchanged.
// SESS > 1 (Model::ds_slots = --s2-sessions 2..32, GR8_DS_PREFILL_MIN >= 2): S2 slot q owns slab q of every attention / index / Engram
// state (ds41_attn_ops.cuh, ds41_idx_ops.cuh, ds41_engram_ops.cuh) and, with DSpark, of the draft layers' key rings. A bind clears only that
// slot's slabs (and its cached drafts); a step's rows carry their slot in StepIn::sess; a MAIN step of ns > 1 sessions has the same rows per
// session (1 at MTP 0, the verify's k + 1 with DSpark; the prompt runs as W-row prefill, its last token is the session's first MAIN row) and
// launches the ns-session graph (Model::ds_ns = ns). The pool holds ds_slots x cap rows; one run takes at most cap.
#pragma once
#if G6_MODEL_DS41
#include "service.h"
#include "ds41_dspark_round.cuh"
#include "ds41_prefix_cache.cuh"
#include "../ds41_fabric_error.h"
#include "ds41_prefill.h"
#include "ds41_prefill_lanes.h"
#include "ds41_step_layout.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
namespace g6s2 {
// native_inputs fills the host S2In of a MAIN step: it must hold every S2 session (SESS=32; r2-d5 window crashed at 9 with S2N 8)
static_assert(g6rt::S2N >= int(MAX_SESSIONS), "S2In holds fewer sessions than the S2 scheduler admits");
struct Ds41Context {
    g6rt::Model& main;
    cudaStream_t stream;
    g6rt::StepIn* input;
    g6rt::StepOut* output;
    // the step's collective sizes and the fabric / CP / GR8_QA_SPLIT enqueues (g6m4.cu: the same lambda the GLM path hands NativeContext)
    std::function<void(const g6rt::StepIn&)> enqueue;
    // the fabric, CP and GR8_QA_SPLIT error words, ORed (0 = clean); read after the stream is synchronized
    std::function<unsigned()> errors;
    // launch the step: the captured graph of context bucket k and si.M rows (g6m4.cu launch_main), or without --graph the eager step at si.context
    std::function<void(int k, const g6rt::StepIn& si)> launch;
    // DSpark MTP (GR8_DS_MTP): the draft Model and its launch; null = MTP 0 (sc.mtp == 0 exactly when this is null)
    Ds41Dspark* dsp = nullptr;
    // W-row prefill (GR8_DS_PREFILL_W): enqueue and launch one chunk, no sync (ds41_prefill.h); null = off (prefill_min == 0 exactly then)
    Ds41PrefillFn prefill;
    // After the chunk's stream fence: diagnostic counters and per-layer error words.
    std::function<void(const Ds41PrefillChunk&)> after_prefill;
    // GR8_DS_PF_LANES=2 (ds41_prefill_lanes.h): enqueue and launch lane A (the first chunk, main fabric) and lane B (the second, lane B's
    // fabric) of one quantum, no sync; lane B's fabric error word (0 = clean, read after the fence); lane B's counters after the fence.
    // All null = one lane (Config::two_lanes false exactly then).
    std::function<void(const Ds41PrefillChunk&, const Ds41PrefillChunk&)> prefill2;
    std::function<unsigned()> errors2;
    std::function<void(const Ds41PrefillChunk&)> after_prefill2;
    // F12: one ragged W pass over 2..8 bounded segments, one main fabric sequence; null until stage 2 wires the device path.
    Ds41PrefillPackFn prefill_pack;
    std::function<void(const Ds41PrefillPack&)> after_prefill_pack;
    Ds41PrefixCache* ckpt = nullptr; // optional, owned by the caller
};
inline void serve_ds41(Ds41Context c, const BootstrapConfig& bc, int chunk) {
    using namespace g6rt;
    const Config& sc = bc.scheduler;
    const int slots = c.main.ds_slots;
    require(sc.limit() == unsigned(slots) && sc.mtp <= 5 && (sc.mtp == 0) == (c.dsp == nullptr) && sc.prefill_rows > 0,
            "DS41 serve is --s2-sessions sessions at MTP 0, or at MTP 1..5 with the DSpark draft Model");
    require(slots == 1 || (bc.prefill_min >= 2 && sc.pool_rows % (unsigned(slots) * PAGE) == 0),
            "DS41 SESS > 1 needs GR8_DS_PREFILL_MIN >= 2 and a pool of sessions x whole pages");
    require((c.prefill != nullptr) == (bc.prefill_min != 0), "DS41 serve: the prefill callee and prefill_min are set together");
    require(!c.prefill || ((!sc.two_lanes || c.prefill2) && sc.prefill_rows <= uint32_t(c.main.hc_rows)),
            "DS41 prefill is one lane of at most W rows (Config::prefill_rows <= GR8_DS_PREFILL_W), or two lanes with the lane callee");
    require(!c.prefill2 || (c.prefill && c.errors2 && sc.two_lanes && sc.prefill_rows == uint32_t(c.main.hc_rows) &&
                            sc.decode_prefill_rows >= 1 && sc.decode_prefill_rows <= sc.prefill_rows),
            "DS41 two-lane prefill needs the one-lane callee, lane B's error word, two_lanes at W rows and the decode cap");
    std::unique_ptr<Ds41Round> round;
    if (c.dsp) round = std::make_unique<Ds41Round>(c.main, *c.dsp, c.stream, int(sc.mtp));
    const unsigned pages = sc.pool_rows / PAGE;
    const unsigned slot_pages = pages / unsigned(slots);   // SESS > 1: one run fits one slot's slabs (cap rows); slots == 1: the pool
    RunPool pool(pages, pages, sc.pool_rows);
    Ds41PrefixCache* const ck = c.ckpt;
    std::mutex ck_mu; // head admission/status can read while compute captures/evicts
    std::atomic<uint64_t> claimed{0};
    std::array<Run, MAX_SESSIONS> bound{};
    std::array<bool, MAX_SESSIONS> bound_live{};
    LpStep lp{};
    int lp_rows = 0;
    auto fence = [&] { G6RT_CK(cudaStreamSynchronize(c.stream)); };
    NativeHooks h;
    h.pool.free_pages = [&] { return pool.free_pages(); };
    h.pool.room = [&] { return std::min(pool.room(), slot_pages); };
    h.pool.reserve = [&](uint64_t n, uint64_t worst) {
        require(worst <= uint64_t(slot_pages) * PAGE, "DS41 run larger than one session's slab");
        return pool.reserve(n, worst);
    };
    h.pool.release = [&](const Run& r) { pool.release(r); };
    h.pool.fits = [&](uint64_t n, uint64_t worst) { return worst <= uint64_t(slot_pages) * PAGE && pool.fits(n, worst); };
    h.pool.digest = [&]() -> uint64_t {
        if (!ck) return pool.digest();
        std::lock_guard<std::mutex> l(ck_mu);
        return pool.digest() ^ (ck->digest() * 0x9E3779B97F4A7C15ull);
    };
    if (ck) {
        h.pool.retain = [&](const Run& r, const PrefixReuse& st, const CacheNamespace& ns) {
            int q = -1;
            for (int i = 0; i < slots; ++i)
                if (bound_live[size_t(i)] && bound[size_t(i)] == r) { q = i; break; }
            require(q >= 0, "DS41 prefix retain: run has no bound slot");
            // DSpark's final MAIN verify may finish without a following DRAFT/comp_commit;
            // only st.valid (which includes draft_end) is safe to snapshot then.
            const uint32_t p = st.history.empty() ? 0u : uint32_t(c.dsp
                ? st.valid : std::min<int>(st.main_end, int(st.history.size()) - 1));
            if (p) {
                std::lock_guard<std::mutex> l(ck_mu);
                const uint64_t gen = ck->capture(q, p, st.history, ns, c.stream);
                if (gen) fprintf(stderr, "# rank %d ds41 ckpt capture gen %llu slot %d pos %u bytes %zu used %zu/%zu\n",
                                 c.main.rank, (unsigned long long)gen, q, p, ck->last.bytes,
                                 ck->used_bytes(), ck->budget_bytes());
            }
            bound_live[size_t(q)] = false;
            pool.release(r); // the snapshot owns separate device storage, not the run's pages
        };
        h.pool.match = [&](const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit, const CacheNamespace& ns) {
            std::lock_guard<std::mutex> l(ck_mu);
            const Ds41PrefixHit f = ck->match(prompt, limit, ns);
            PoolHit hit; hit.lcp = f.lcp; hit.why = f.why;
            if (f.hit && pool.fits(rows, worst)) { hit.skip = int(f.position); hit.age = f.generation; }
            else if (f.hit) hit.why = "room";
            return hit;
        };
        h.pool.take = [&](const PoolHit& hit, uint64_t rows, uint64_t worst) {
            claimed.store(hit.age);
            return pool.reserve(rows, worst);
        };
        h.pool.tiers = [&] {
            std::lock_guard<std::mutex> l(ck_mu);
            return std::vector<g6kv_tier_totals>{ck->tier_stat()};
        };
    }
    // a request starts at position 0 on the pool's first pages (no reuse, no adoption of a retired run): reset() leaves the first
    // request's zeroed buffers alone and clears the attention and index state before every later one. No step is in flight here.
    h.bind = [&](const Grant& g) {
        if (slots > 1) {   // a fresh run on slot q: clear that slot's slabs only (the other sessions keep theirs; no step is in flight)
            const int q = int(g.key.slot);
            require(q >= 0 && q < slots && (!g.reuse || ck), "DS41 serve requires a fresh run or prefix snapshot on a session slot");
            fence();
            require(ds41_attn_clear(&c.main, c.stream, q) == 0, "DS41 bind: attention slot clear failed");
            require(ds41_idx_clear(&c.main, c.stream, q) == 0, "DS41 bind: index slot clear failed");
            if (round) round->reset(q);           // DSpark: that slot's cached drafts and its draft key-ring slab
            fence();
            lp_rows = 0;
            bound[size_t(q)] = g.run; bound_live[size_t(q)] = true;
            if (g.reuse) {
                const uint64_t gen = claimed.exchange(0);
                require(gen != 0, "DS41 prefix hit without a claimed entry");
                std::lock_guard<std::mutex> l(ck_mu);
                const Ds41PrefixEntry* e = ck->find(gen);
                require(e && e->position == g.reuse, "DS41 prefix claim differs from granted skip");
                ck->restore(gen, q, c.stream);
                fprintf(stderr, "# rank %d ds41 ckpt restore gen %llu slot %d pos %u bytes %zu hits %llu\n",
                        c.main.rank, (unsigned long long)gen, q, g.reuse, ck->last.bytes,
                        (unsigned long long)ck->hits());
            }
            return;
        }
        require(g.key.slot == 0 && (!g.reuse || ck) && g.run.first_page == 0, "DS41 serve requires an identity run on slot 0");
        const auto list = pool.list(g.run);
        for (size_t i = 0; i < list.size(); ++i) require(list[i] == int(i), "DS41 run is not contiguous");
        fence();
        reset(c.main, c.stream);
        fence();
        if (round) round->reset();
        fence();
        lp_rows = 0;
        bound[0] = g.run; bound_live[0] = true;
        if (g.reuse) {
            const uint64_t gen = claimed.exchange(0);
            require(gen != 0, "DS41 prefix hit without a claimed entry");
            std::lock_guard<std::mutex> l(ck_mu);
            const Ds41PrefixEntry* e = ck->find(gen);
            require(e && e->position == g.reuse, "DS41 prefix claim differs from granted skip");
            ck->restore(gen, 0, c.stream);
            fprintf(stderr, "# rank %d ds41 ckpt restore gen %llu slot 0 pos %u bytes %zu hits %llu\n",
                    c.main.rank, (unsigned long long)gen, g.reuse, ck->last.bytes,
                    (unsigned long long)ck->hits());
        }
    };
    h.retire = [&](const Slot&) { fence(); };
    h.fence = fence;
    h.dispatch.bucket = [&](bool, int ctx) { return bucket_of(c.main, ctx); };
    h.dispatch.mode = [&](int, int next) { fence(); require(slots > 1 || next == -1 || next == 0, "DS41 non-solo mode"); };
    h.dispatch.launch = [&](const Launch& l) {
        require(slots > 1 || (l.solo == 0 && l.step.sess.size() == 1), "DS41 launch is not solo");
        if (l.step.kind == s2::Step::DRAFT && round) return round->draft_step(l);
        if (l.step.kind == s2::Step::CHAIN && round) return round->chain_step(l);
        if (l.step.kind != s2::Step::MAIN) return std::vector<int>(size_t(l.step.M()), 0);
        StepIn si{}; S2In h2{};
        native_inputs(l, si, h2);
        if (slots > 1) {   // each row's slot; ns sessions of rps rows (ns > 1: the captured ns-session graphs, rps = 1 at MTP 0, 1..k + 1 with DSpark)
            if (const char* e = ds41_session_rows(h2, slots, int(sc.mtp), bool(round), si)) require(false, e);
            c.main.ds_ns = h2.ns;
        }
        *c.input = si;
        c.main.ds_moe_decode_step = l.step.scheduler_decode;
        c.enqueue(si);
        c.launch(l.buckets[0], si);
        fence();
        if (const unsigned e = c.errors()) {      // names the code, the stuck collective and the peer (first Engram light, 10-25: no word reached any log)
            std::vector<int> ids;
            for (const auto& y : c.main.layers) ids.push_back(y.l);
            require(false, ds41::fabric_error_text("DS41 serve fabric error", e, ids, c.main.ecoll, ds41::engram::kLayerIds).c_str());
        }
        const auto& out = *c.output;
        require(!out.attn_err, "DS41 step attention error");
        lp_rows = 0;
        if (std::any_of(l.samples.begin(), l.samples.end(), [](const Sampling& s) { return s.lp >= 0 || s.plp >= 0; })) {
            lp_fetch(c.main, out, si, lp); lp_rows = si.M;
        }
        return std::vector<int>(out.tok, out.tok + si.M);
    };
    h.dispatch.logprob = [&](int row, bool target, int k) {
        require(row >= 0 && row < lp_rows, "DS41 logprob row outside main step");
        return lp_row(lp, row, target, k);
    };
    h.dispatch.prefill = [&](const Slot&, const Payload&, uint32_t, uint32_t) {
        require(false, "DS41 serve: prefill is not built (every prompt runs as decode steps)");
    };
    auto capture_prompt_end = [&](const Slot& s, const Payload& p, uint32_t end) {
        if (!ck || p.prompt.empty() || end != p.prompt.size() - 1) return;
        const CacheNamespace ns = steering::cache_namespace(s.req.steering);
        std::lock_guard<std::mutex> l(ck_mu);
        const uint64_t gen = ck->capture(int(s.key.slot), end, p.prompt, ns, c.stream);
        if (gen) fprintf(stderr, "# rank %d ds41 ckpt capture gen %llu slot %u pos %u bytes %zu used %zu/%zu\n",
                         c.main.rank, (unsigned long long)gen, unsigned(s.key.slot), end,
                         ck->last.bytes, ck->used_bytes(), ck->budget_bytes());
    };
    if (c.prefill && !c.prefill2)   // one chunk: ONE id of the main fabric sequence (Dispatch::prefill), fenced like a step; no head, so no output to read
        h.dispatch.prefill_seq = [&](const Slot& s, const Payload& p, uint32_t first, uint32_t rows, uint64_t sequence) {
            c.main.ds_moe_decode_step = false;
            require(!p.decode_prompt, "DS41 prefill of a decode_prompt session");
            const Ds41PrefillChunk chunk = ds41_make_chunk(p.prompt, first, rows, sc.prefill_rows, sequence, s.req.id, s.key.slot, s.run);
            c.prefill(chunk);
            fence();
            if (const unsigned e = c.errors()) {
                std::vector<int> ids;
                for (const auto& y : c.main.layers) ids.push_back(y.l);
                require(false, ds41::fabric_error_text("DS41 prefill fabric error", e, ids, c.main.ecoll, ds41::engram::kLayerIds).c_str());
            }
            int layer = -1;
            if (const unsigned e = attn_errors_at(c.main, &layer))
                require(false, ("DS41 prefill attention error layer " + std::to_string(layer) + " word " + std::to_string(e)).c_str());
            if (c.after_prefill) c.after_prefill(chunk);
            capture_prompt_end(s, p, first + rows);
        };
    ds41::lanes::Counter lane_b;   // lane B's fabric sequence: one id per lane-B chunk
    if (c.prefill2)   // a quantum of W + 1 .. 2W rows: lane A + lane B, ONE fence; up to W rows: the one-lane callee, as above
        h.dispatch.prefill_seq = [&](const Slot& s, const Payload& p, uint32_t first, uint32_t rows, uint64_t sequence) {
            c.main.ds_moe_decode_step = false;
            require(!p.decode_prompt, "DS41 prefill of a decode_prompt session");
            const Ds41LaneChunks q = ds41_make_lane_chunks(p.prompt, first, rows, sc.prefill_rows, sc.two_lanes, sequence, lane_b,
                                                           s.req.id, s.key.slot, s.run);
            if (q.b.rows) c.prefill2(q.a, q.b); else c.prefill(q.a);
            fence();
            for (int lane = 0; lane < (q.b.rows ? 2 : 1); ++lane)
                if (const unsigned e = lane ? c.errors2() : c.errors()) {
                    std::vector<int> ids;
                    for (const auto& y : c.main.layers) ids.push_back(y.l);
                    require(false, ds41::fabric_error_text(lane ? "DS41 prefill lane B fabric error" : "DS41 prefill fabric error", e, ids,
                                                           c.main.ecoll, ds41::engram::kLayerIds).c_str());
                }
            int layer = -1;   // the per-layer attention error words are shared by both lanes (ORed)
            if (const unsigned e = attn_errors_at(c.main, &layer))
                require(false, ("DS41 prefill attention error layer " + std::to_string(layer) + " word " + std::to_string(e)).c_str());
            if (c.after_prefill) c.after_prefill(q.a);
            if (q.b.rows && c.after_prefill2) c.after_prefill2(q.b);
            capture_prompt_end(s, p, first + rows);
        };
    if (c.prefill_pack)
        h.dispatch.prefill_pack_seq = [&](const std::vector<PrefillPart>& parts, uint64_t sequence) {
            c.main.ds_moe_decode_step = false;
            std::vector<Ds41PrefillChunk> chunks; chunks.reserve(parts.size());
            for (const auto& p : parts)
                chunks.push_back(ds41_make_chunk(p.payload->prompt, p.first, p.rows, sc.prefill_rows,
                                                  sequence, p.slot->req.id, p.slot->key.slot, p.slot->run));
            const Ds41PrefillPack pack = ds41_make_pack(chunks, sc.prefill_rows, unsigned(slots));
            c.prefill_pack(pack);
            fence(); // one W pass and one fence, not N solo W steps
            if (const unsigned e = c.errors()) {
                std::vector<int> ids;
                for (const auto& y : c.main.layers) ids.push_back(y.l);
                require(false, ds41::fabric_error_text("DS41 ragged prefill fabric error", e, ids, c.main.ecoll, ds41::engram::kLayerIds).c_str());
            }
            int layer = -1;
            if (const unsigned e = attn_errors_at(c.main, &layer))
                require(false, ("DS41 ragged prefill attention error layer " + std::to_string(layer) + " word " + std::to_string(e)).c_str());
            if (c.after_prefill_pack) c.after_prefill_pack(pack);
            for (const auto& part : parts)
                capture_prompt_end(*part.slot, *part.payload, part.first + part.rows);
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
