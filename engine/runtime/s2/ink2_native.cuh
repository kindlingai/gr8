// Inkling-2-Small behind the S2 service (service.h): the module hooks over ink2rt::Engine (../ink2_step.cuh).
//
// MTP 0: a step is MAIN rows only, one row a decoding session or a run of prompt rows; every row is kept. A prompt
// runs through the prefill hook in steps of at most the engine's rows (with the head skipped). Each session slot owns
// its pages of the full-attention layers (the engine's page table gives slot s pages s * max_pps ..). With
// --reuse-prefix / --kv-store, StashPool + KvStateTiers snapshot and restore full 64-token global-attention pages and
// boundary state (sliding-window rings, short-convolution histories and partial-page staging buffers) across turns and
// persistent SSD restarts. Every rank runs the same calls in the same order, so the ranks' fabric steps line up.
#ifndef G6_S2_INK2_NATIVE_CUH
#define G6_S2_INK2_NATIVE_CUH
#include "service.h"
#include "kv_state_desc.h"
#include "kv_state_mover.cuh"
#include "../ink2_step.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <unistd.h>

namespace g6s2 {
struct Ink2Context {
    ink2rt::Engine& engine;
    std::function<void(const ink2::RowsHost&, const int* toks, const ink2rt::RowSample*, std::vector<ink2rt::RowOut>&)> step;
    int rank = 0;
    bool reuse = false;
    std::vector<std::string> kv_tiers;
    std::string kv_keep = "all";
    int kv_sys_after = 2;
    double kv_wlim = 0.0;
    std::string set_id;
};

inline void serve_ink2(Ink2Context c, const BootstrapConfig& bc, int chunk) {
    const Config& sc = bc.scheduler;
    const int slots = c.engine.o.slots, max_rows = c.engine.o.max_rows;
    require(sc.mtp == 0 && sc.max_sessions == unsigned(slots) && sc.max_rows <= unsigned(max_rows),
            "Inkling-2-Small serves --s2-sessions sessions at MTP 0 in steps of at most the engine's rows");
    require(sc.pool_rows % (unsigned(slots) * 64u) == 0 && sc.pool_rows / unsigned(slots) <= unsigned(c.engine.o.cap),
            "Inkling-2-Small: the pool is sessions x whole pages within the cap");
    const unsigned pages = sc.pool_rows / 64u;
    const unsigned max_pps = unsigned(c.engine.max_pages_per_slot());
    RunPool pool(pages, pages, sc.pool_rows);
    std::vector<ink2rt::RowOut> out;
    std::vector<int> targets;
    int lp_rows = 0;

    std::unique_ptr<KvStateTiers> tiers;
    std::unique_ptr<StashPool> stash;
    bool kv_index = false, store_shared = false, did_drain = false;
    std::mutex ck_mu;
    struct Claimed { uint64_t age = 0; uint32_t tier = 0; int skip = 0; };
    std::deque<Claimed> claimed;
    struct StoreSlot {
        uint32_t hits = 0, pages = 0, earned = 0, ckpt = 0;
        uint64_t id = 0, prompt_id = 0;
        int pf_end = 0;
        int64_t sys = -1;
        bool restored_end = false;
    };
    std::array<StoreSlot, MAX_SESSIONS> store_slots{};
    std::array<Run, MAX_SESSIONS> bound{};
    std::array<bool, MAX_SESSIONS> bound_live{};
    std::array<int64_t, MAX_SESSIONS> incoming_sys{};

    if (c.reuse) {
        std::vector<Ink2KvLayer> kl;
        for (const auto& p : c.engine.kv_ptrs(0)) {
            kl.push_back(Ink2KvLayer{p.global, p.pages, p.staging, p.ring, p.hist_k, p.hist_v, p.hist_a, p.hist_m});
        }
        KvStateDesc desc = ink2_kv_state(kl, uint32_t(slots), max_pps, uint32_t(c.engine.rank_nk(0)),
                                         uint32_t(c.engine.c.H), uint32_t(c.engine.c.window), uint32_t(c.engine.max_rank_nk()));
        tiers = std::make_unique<KvStateTiers>(desc);
        stash = std::make_unique<StashPool>(uint32_t(c.kv_sys_after), 64, desc.state_blocks(), false);
        stash->set_rules(false, true, 0);
        std::vector<std::string> specs = c.kv_tiers;
        if (specs.empty()) specs.push_back("mem:,size=512M,keep=" + c.kv_keep);
        const std::string exe = exe_stamp();
        for (size_t i = 0; i < specs.size(); ++i) {
            const auto t = TierSpec::parse(specs[i], false);
            std::string spec = t.store, index;
            if (t.ssd()) {
                const auto dir = t.store.substr(4) + "/r" + std::to_string(c.rank);
                if (!t.persist && std::filesystem::exists(dir)) std::filesystem::remove_all(dir);
                std::filesystem::create_directories(dir);
                spec = "ssd:" + dir + "/blocks";
                if (t.persist && !c.set_id.empty()) index = dir;
            }
            if (t.spare_ram) spec = "ram:" + std::to_string(t.spare_ram_floor) + "+" + spec;
            const std::string name = "ink2-" + c.set_id + "-" + std::to_string(getpid()) +
                                     "-r" + std::to_string(c.rank) + "-t" + std::to_string(i);
            const auto blocks = t.group.empty()
                ? tiers->add(spec, name, t.bytes)
                : tiers->add_group(spec, name, t.group, unsigned(c.engine.o.nr), unsigned(c.rank), t.bytes, t.cap, t.weight);
            const std::string keep = t.rule.empty() ? c.kv_keep : t.rule;
            const bool whole = !index.empty() && keep != "system";
            store_shared |= keep == "system" || keep == "all";
            kv_index |= !index.empty();
            const std::string compat = "ink2 set " + c.set_id + " nr " + std::to_string(c.engine.o.nr) +
                                       " slots " + std::to_string(slots) + " pages " + std::to_string(max_pps) +
                                       " layers " + c.engine.o.layers +
                                       " format " + std::to_string(desc.format_hash()) +
                                       " state " + std::to_string(desc.state_blocks()) +
                                       " blocks " + std::to_string(blocks) +
                                       " engine " + exe;
            stash->add_tier(blocks, (keep == "system" ? 1 : 2) | (whole ? 4 : 0),
                            index.empty() ? nullptr : index.c_str(), compat.c_str());
            stash->set_policy(uint32_t(i), 0, keep == "sessions" ? 1 : 0, desc.slot_bytes());
            fprintf(stderr, "# rank %d: s2 store %zu: %s, %llu blocks of %zu B, keep %s%s\n",
                    c.rank, i, tiers->describe(i).c_str(), (unsigned long long)blocks, desc.slot_bytes(),
                    keep.c_str(), index.empty() ? "" : ", indexed");
        }
        stash->tiers_loaded();
        stash->write_limit(uint64_t(std::llround(c.kv_wlim * 1000)));
    }

    const auto state_pages = [&](unsigned q, unsigned pos) {
        std::vector<uint32_t> v(pos / 64u);
        for (unsigned i = 0; i < v.size(); ++i) v[i] = pack_page_handle(q, i, max_pps);
        return v;
    };
    const auto drain_store = [&]() {
        if (!stash) return;
        const auto io = stash->io();
        if (!io.empty()) did_drain = true;
        tiers->run(io, 0);
        stash->io_done();
    };
    const auto run_rows = [&](int slot, const int* toks, int first, int n, const ink2rt::RowSample* sm) {
        ink2::RowsHost rh;
        rh.add(slot, first, n);
        c.step(rh, toks, sm, out);
    };

    NativeHooks h;
    h.pool.pool = &pool;
    h.pool.max_run_pages = pages / unsigned(slots);   // one run fits one slot's pages
    if (stash) {
        h.checkpoint = [&](const g6rt::ServeReq& r) -> uint32_t {
            if (!store_shared || r.prompt.empty() || r.steering.enabled) return 0;
            const uint64_t top = std::min<uint64_t>(r.prompt.size() - 1, r.sys < 0 ? r.prompt.size() - 1 : uint64_t(r.sys));
            return uint32_t(top / 64u * 64u);
        };
        h.pool.digest = [&]() -> uint64_t {
            std::lock_guard<std::mutex> lk(ck_mu);
            return pool.digest() ^ (stash->digest() * 0x9E3779B97F4A7C15ull);
        };
        h.pool.match = [&](const std::vector<int>& prompt, uint64_t rows, uint64_t worst, int limit,
                           const g6rt::CacheNamespace& ns, uint32_t r) -> PoolHit {
            std::lock_guard<std::mutex> lk(ck_mu);
            const g6kv_hit f = stash->match(prompt, limit, ns);
            PoolHit hit;
            hit.lcp = f.lcp;
            hit.why = f.why;
            if (f.stored && f.skip > 0) {
                if (pool.fits(rows, worst, r)) {
                    hit.skip = f.skip;
                    hit.age = f.age;
                    hit.stored = true;
                    hit.tier = f.tier;
                } else {
                    hit.why = "room";
                }
            }
            return hit;
        };
        h.pool.take = [&](const PoolHit& hit, uint64_t rows, uint64_t worst, uint32_t r) -> Run {
            std::lock_guard<std::mutex> lk(ck_mu);
            if (hit.skip > 0) claimed.push_back(Claimed{hit.age, hit.tier, hit.skip});
            return pool.reserve(rows, worst, r);
        };
        h.pool.tiers = [&]() -> std::vector<g6kv_tier_totals> {
            std::lock_guard<std::mutex> lk(ck_mu);
            std::vector<g6kv_tier_totals> v;
            for (size_t i = 0; i < tiers->size(); ++i) v.push_back(stash->tier_stats(uint32_t(i)));
            return v;
        };
        h.pool.retain = [&](const Run& r, const g6rt::PrefixReuse& st, const g6rt::CacheNamespace& ns) {
            int q = -1;
            for (int i = 0; i < slots; ++i)
                if (bound_live[size_t(i)] && bound[size_t(i)] == r) { q = i; break; }
            require(q >= 0, "Inkling-2-Small retain: run has no bound slot");
            const uint32_t p = st.history.empty() ? 0u : uint32_t(std::max(0, std::min<int>(st.main_end, int(st.history.size()) - 1)));
            if (p > 0) {
                c.engine.stage_partial_page(q, p, true);
                std::lock_guard<std::mutex> lk(ck_mu);
                StoreSlot& ss = store_slots[size_t(q)];
                if (p > ss.earned) {
                    stash->earn(p / 64u - ss.earned / 64u + tiers->desc().state_blocks());
                    ss.earned = p;
                }
                const bool keep_from = !ss.restored_end && ss.prompt_id != 0 && ss.prompt_id == ss.id;
                const uint64_t id = stash->capture(st.history, int32_t(p), int32_t(p), ss.pf_end,
                                                   state_pages(unsigned(q), p), pack_state_handle(uint32_t(q), p),
                                                   ss.hits, ns, ss.id, ss.pages, keep_from, 0);
                drain_store();
                if (id) {
                    ss.pages = uint32_t(stash->info(id).pages);
                    ss.id = id;
                    fprintf(stderr, "# rank %d ink2 kv capture id %llu slot %d pos %u (request end)\n",
                            c.rank, (unsigned long long)id, q, p);
                }
            }
            bound_live[size_t(q)] = false;
            store_slots[size_t(q)] = StoreSlot{};
            pool.release(r);
        };
        h.drained = [&]() -> bool {
            const bool d = did_drain;
            did_drain = false;
            return d;
        };
        h.bind_request = [&](unsigned slot, const g6rt::ServeReq& r) {
            require(slot < MAX_SESSIONS, "Inkling-2-Small bind_request: slot outside the sessions");
            incoming_sys[slot] = r.sys;
            std::lock_guard<std::mutex> lk(ck_mu);
            stash->count(r.prompt, r.sys, g6rt::steering::cache_namespace(r.steering), !claimed.empty());
            stash->io_done();
        };
    }

    h.bind = [&](const Grant& g) {
        const int q = int(g.key.slot);
        require(q >= 0 && q < slots, "Inkling-2-Small bind: slot outside the sessions");
        require((g.reuse == 0 && g.restored == 0) || stash, "Inkling-2-Small serve: prefix reuse requires --reuse-prefix or --kv-store");
        c.engine.reset_slot(q);
        INK2RT_CK(cudaDeviceSynchronize());
        bound[size_t(q)] = g.run;
        bound_live[size_t(q)] = true;
        if (stash) {
            StoreSlot ss{};
            ss.sys = incoming_sys[size_t(q)];
            ss.ckpt = g.req.ckpt_at > g.reuse ? g.req.ckpt_at : 0;
            if (g.reuse > 0) {
                std::lock_guard<std::mutex> lk(ck_mu);
                require(!claimed.empty(), "Inkling-2-Small bind: prefix hit without a claimed entry");
                const Claimed cl = claimed.front();
                claimed.pop_front();
                require(uint32_t(cl.skip) == g.reuse && stash->has(cl.age), "Inkling-2-Small bind: claimed entry differs from grant");
                const auto meta = stash->info(cl.age);
                const auto r = stash->restore(cl.tier, cl.age, state_pages(unsigned(q), g.reuse), pack_state_handle(uint32_t(q), g.reuse));
                require(r.valid == int32_t(g.reuse), "Inkling-2-Small bind: restored position differs from grant");
                drain_store();
                ss.hits = r.hits + 1;
                ss.pages = uint32_t(r.full_pages);
                ss.earned = g.reuse;
                ss.id = cl.age;
                ss.pf_end = meta.pf_end;
                ss.restored_end = g.reuse != uint32_t(meta.pf_end);
                fprintf(stderr, "# rank %d ink2 kv restore id %llu tier %u slot %d pos %u pages %zu hits %u\n",
                        c.rank, (unsigned long long)cl.age, cl.tier, q, g.reuse, r.full_pages, ss.hits);
            }
            if (g.reuse > 0) c.engine.stage_partial_page(q, g.reuse, false);
            store_slots[size_t(q)] = ss;
        }
    };
    h.retire = [&](const Slot& s) {
        if (!stash && int(s.key.slot) >= 0 && int(s.key.slot) < slots) bound_live[s.key.slot] = false;
        ink2rt::ck(cudaDeviceSynchronize(), "retire");
    };
    h.fence = []() { ink2rt::ck(cudaDeviceSynchronize(), "fence"); };
    h.dispatch.bucket = [](bool, int) { return 0; };
    h.dispatch.mode = [](int, int) {};
    h.dispatch.launch = [&](const Launch& l) -> std::vector<int> {
        const s2::Step& sp = l.step;
        const int M = sp.M();
        if (sp.kind != s2::Step::MAIN) return std::vector<int>(size_t(M), 0);
        require(M >= 1 && M <= max_rows && l.samples.size() >= size_t(M), "Inkling-2-Small launch: rows");
        ink2::RowsHost rh;
        std::vector<int> toks(static_cast<size_t>(M));
        std::vector<ink2rt::RowSample> sm(static_cast<size_t>(M));
        targets.assign(static_cast<size_t>(M), -1);
        rh.slot.resize(size_t(M)); rh.pos.resize(size_t(M)); rh.run0.resize(size_t(M));
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            require(sp.sess[j] >= 0 && sp.sess[j] < slots, "Inkling-2-Small launch: session outside the slots");
            for (int r = sp.row0[j]; r < sp.row0[j] + sp.rows[j]; ++r) {
                require(r >= 0 && r < M, "Inkling-2-Small launch: row outside the step");
                const s2::Row& x = sp.r[size_t(r)];
                const bool cont = r > sp.row0[j] && x.pos == sp.r[size_t(r) - 1].pos + 1;
                rh.slot[size_t(r)] = sp.sess[j]; rh.pos[size_t(r)] = x.pos; rh.run0[size_t(r)] = cont ? rh.run0[size_t(r) - 1] : r;
                rh.max_pos = std::max(rh.max_pos, x.pos);
                toks[size_t(r)] = x.tok; targets[size_t(r)] = x.tgt;
                const Sampling& s = l.samples[size_t(r)];
                sm[size_t(r)] = ink2rt::RowSample{s.inv_temp, s.top_p, s.seed, uint32_t(x.spos)};
            }
        }
        c.step(rh, toks.data(), sm.data(), out);
        lp_rows = M;
        std::vector<int> ids(static_cast<size_t>(M));
        for (int r = 0; r < M; ++r) ids[size_t(r)] = out[size_t(r)].tok;
        return ids;
    };
    // a row's entry from its merged candidates: a token outside them has no rank, and the smallest candidate bounds it
    h.dispatch.logprob = [&](int row, bool target, int k) -> g6rt::LpRow {
        require(row >= 0 && row < lp_rows, "Inkling-2-Small logprob row outside the main step");
        const ink2rt::RowOut& o = out[size_t(row)];
        const int id = target ? targets[size_t(row)] : o.tok;
        g6rt::LpRow e;
        e.n = std::max(0, std::min({k, int(g6rt::LP_MAX), int(o.top.size())}));
        for (int i = 0; i < e.n; ++i) { e.id[i] = o.top[size_t(i)].id; e.tlp[i] = o.top[size_t(i)].logit - o.lse; }
        e.lp = o.top.empty() ? 0.f : o.top.back().logit - o.lse;
        for (size_t i = 0; i < o.top.size(); ++i)
            if (o.top[i].id == id) { e.lp = o.top[i].logit - o.lse; e.rank = i < size_t(g6rt::LP_MAX) ? int(i) : -1; break; }
        return e;
    };
    h.dispatch.prefill = [&](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
        const int q = int(s.key.slot);
        require(size_t(first) + rows <= p.prompt.size() && q >= 0 && q < slots, "Inkling-2-Small prefill: rows outside the prompt");
        uint32_t at = first;
        const uint32_t limit = first + rows;
        while (at < limit) {
            uint32_t step_lim = std::min<uint32_t>(limit, at + uint32_t(max_rows));
            if (stash) {
                const uint32_t ck = store_slots[size_t(q)].ckpt;
                if (ck > at && ck < step_lim) step_lim = ck;   // stop on the system checkpoint boundary so promote captures exact state
            }
            const int n = int(step_lim - at);
            run_rows(q, &p.prompt[at], int(at), n, nullptr);
            at = step_lim;
            if (stash && store_slots[size_t(q)].ckpt && at == store_slots[size_t(q)].ckpt) {
                c.engine.stage_partial_page(q, at, true);
                std::lock_guard<std::mutex> lk(ck_mu);
                StoreSlot& ss = store_slots[size_t(q)];
                if (at > ss.earned) {
                    stash->earn(at / 64u - ss.earned / 64u + tiers->desc().state_blocks());
                    ss.earned = at;
                }
                stash->promote(p.prompt, int32_t(at), int32_t(at), int32_t(at), state_pages(unsigned(q), at),
                               pack_state_handle(uint32_t(q), at), at, ss.sys);
                drain_store();
                ss.ckpt = 0;
            }
        }
        if (stash && !p.prompt.empty() && limit == p.prompt.size() - 1 && limit > 0) {
            c.engine.stage_partial_page(q, limit, true);
            const g6rt::CacheNamespace ns = g6rt::steering::cache_namespace(s.req.steering);
            std::lock_guard<std::mutex> lk(ck_mu);
            StoreSlot& ss = store_slots[size_t(q)];
            if (limit > ss.earned) {
                stash->earn(limit / 64u - ss.earned / 64u + tiers->desc().state_blocks());
                ss.earned = limit;
            }
            ss.pf_end = int(limit);
            const bool keep_from = ss.id != 0;   // keep an earlier restored prefix entry intact when extending it
            const uint64_t id = stash->capture(p.prompt, int32_t(limit), int32_t(limit), ss.pf_end,
                                               state_pages(unsigned(q), limit), pack_state_handle(uint32_t(q), limit),
                                               ss.hits, ns, ss.id, ss.pages, keep_from, 0);
            drain_store();
            if (id) {
                ss.pages = uint32_t(stash->info(id).pages);
                ss.id = id;
                ss.prompt_id = id;
                fprintf(stderr, "# rank %d ink2 kv capture id %llu slot %d pos %u (prompt end)\n",
                        c.rank, (unsigned long long)id, q, limit);
            }
        }
        lp_rows = 0;
    };

    ControlLinks links = bootstrap(bc);
    try {
        if (stash && kv_index) {
            stash->keep_agreed(agree_lines(bc, links, stash->stored_summary()));
            uint64_t n = 0;
            for (size_t i = 0; i < tiers->size(); ++i) n += stash->tier_stats(uint32_t(i)).entries;
            fprintf(stderr, "# rank %d: s2 stores: %llu stored prefixes from earlier runs\n",
                    c.rank, (unsigned long long)n);
        }
        capture_barrier(bc, links);
        Service service(bc, links, std::move(h), chunk);
        service.run();
        if (stash) {
            uint64_t entries = 0, blocks = 0;
            for (size_t i = 0; i < tiers->size(); ++i) {
                const auto t = stash->tier_stats(uint32_t(i));
                entries += t.entries;
                blocks += t.spilled_pages;
            }
            fprintf(stderr, "# rank %d: s2 stores: quit stored %llu entries, %llu blocks written\n",
                    c.rank, (unsigned long long)entries, (unsigned long long)blocks);
        }
        links.shutdown();
    } catch (...) { links.shutdown(); throw; }
}
}   // namespace g6s2
#endif
