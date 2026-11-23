// Qwen3.8-Flash-Next behind the S2 host service (service.h): wraps qw38rt::Engine (../qw38_step.cuh)
// with multi-session scheduling, StashPool + KvStateTiers prefix/session persistence (kv_state_desc.h qw38_kv_state),
// and cross-width live-scale state carry (GR8_KV_REWIDTH via qw38_ckpt_rewidth.h).
#ifndef G6_S2_QW38_NATIVE_CUH
#define G6_S2_QW38_NATIVE_CUH
#include "service.h"
#include "kv_state_desc.h"
#include "kv_state_mover.cuh"
#include "qw38_ckpt_rewidth.h"
#include "../qw38_step.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <unistd.h>

namespace g6s2 {

struct Qw38Context {
    qw38rt::Engine& engine;
    int rank = 0;
    bool reuse = false;
    std::vector<std::string> kv_tiers;
    std::string kv_keep = "all";
    int kv_sys_after = 2;
    double kv_wlim = 0.0;
    std::string set_id;
};

inline void serve_qw38(Qw38Context c, const BootstrapConfig& bc, int chunk) {
    const Config& sc = bc.scheduler;
    const int slots = c.engine.o.slots;
    const int max_rows = c.engine.o.max_rows;
    require(sc.max_sessions == unsigned(slots) && sc.max_rows <= unsigned(max_rows),
            "qw38: --s2-sessions and --max-rows mismatch");
    const unsigned max_pps = unsigned(c.engine.max_pages_per_slot());
    const unsigned pages = sc.pool_rows / geom::KV_PAGE;
    RunPool pool(pages, pages, sc.pool_rows);

    std::vector<qw38rt::RowOut> out;
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
    std::array<int, MAX_SESSIONS> incoming_mrope_delta{};
    std::array<std::vector<g6rt::ImageIn>, MAX_SESSIONS> incoming_img_geom{};

    int32_t* d_ple_ctx = nullptr;
    if (c.reuse) {
        d_ple_ctx = qw38rt::dalloc<int32_t>(size_t(slots) * geom::PLE_NGRAM_CTX);
        std::vector<Qw38FullKvLayer> fl;
        std::vector<Qw38GdnStateLayer> gl;
        for (const auto& y : c.engine.L) {
            const auto& rl = y.rk[0];
            if (y.is_full) {
                fl.push_back(Qw38FullKvLayer{
                    reinterpret_cast<uint8_t*>(rl.kv.kv_pages),
                    reinterpret_cast<uint8_t*>(rl.kv.kv_staging),
                    reinterpret_cast<uint8_t*>(rl.kv.idx_pages),
                    reinterpret_cast<uint8_t*>(rl.kv.idx_staging),
                });
            } else {
                gl.push_back(Qw38GdnStateLayer{
                    reinterpret_cast<uint8_t*>(rl.gdn_st.S),
                    reinterpret_cast<uint8_t*>(rl.gdn_st.conv),
                });
            }
        }
        const int nr = c.engine.o.nr, r = c.rank;
        const uint32_t nh_kv = uint32_t(geom::nh_kv(nr, r));
        const uint32_t nh_v = uint32_t(geom::gdn_vh(nr));
        const uint32_t qkv_l = uint32_t(geom::gdn_qkv_local(nr, r));
        const uint32_t max_nh_kv = uint32_t(geom::nh_kv_max(nr));
        const uint32_t max_qkv = uint32_t((2 * geom::gdn_kh_max(nr) + geom::gdn_vh(nr)) * geom::GDN_DK);

        KvStateDesc desc = qw38_kv_state(
            fl, gl,
            c.engine.has_ple ? reinterpret_cast<uint8_t*>(c.engine.ple_st.conv_hist) : nullptr,
            c.engine.has_ple ? reinterpret_cast<uint8_t*>(d_ple_ctx) : nullptr,
            uint32_t(slots), max_pps, nh_kv, nh_v, qkv_l, max_nh_kv, nh_v, max_qkv);
        tiers = std::make_unique<KvStateTiers>(desc);
        stash = std::make_unique<StashPool>(uint32_t(c.kv_sys_after), geom::KV_PAGE, desc.state_blocks(), false);
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
            const std::string name = "qw38-" + c.set_id + "-" + std::to_string(getpid()) +
                                     "-r" + std::to_string(c.rank) + "-t" + std::to_string(i);
            if (const auto& rw = knob::kv_rewidth(); rw && !index.empty()) {
                Qw38CkptLayout L{nr, r, int(fl.size()), int(gl.size()), desc.slot_bytes()};
                const auto qw_compat = [&](int r_nr, int r_rank, size_t pb, size_t sb, const std::string& set) {
                    return "qw38 set " + set + " nr " + std::to_string(r_nr) + " rank " + std::to_string(r_rank) +
                           " layers " + c.engine.o.layers + " pb " + std::to_string(pb) +
                           " sb " + std::to_string(sb) + " engine " + exe;
                };
                const std::string rw_compat = qw_compat(nr, r, L.page_bytes(), L.state_bytes(), c.set_id);
                std::string why;
                const int64_t carried = qw38_rewidth(index, *rw, rw_compat, qw_compat, L,
                                                     t.bytes / desc.slot_bytes(), geom::KV_PAGE, max_pps, why);
                if (carried > 0) {
                    std::fprintf(stderr, "# rank %d: qw38 ckpt rewidth carried %lld entries from TP=%s\n",
                                 c.rank, (long long)carried, rw->substr(0, rw->find(',')).c_str());
                } else {
                    std::fprintf(stderr, "# rank %d: qw38 ckpt rewidth %s: %s\n",
                                 c.rank, carried == 0 ? "skipped" : "failed", why.c_str());
                }
            }
            const auto blocks = t.group.empty()
                ? tiers->add(spec, name, t.bytes)
                : tiers->add_group(spec, name, t.group, unsigned(nr), unsigned(c.rank), t.bytes, t.cap, t.weight);
            const std::string keep = t.rule.empty() ? c.kv_keep : t.rule;
            const bool whole = !index.empty() && keep != "system";
            store_shared |= keep == "system" || keep == "all";
            kv_index |= !index.empty();
            const std::string compat = "qw38 set " + c.set_id + " nr " + std::to_string(nr) +
                                       " slots " + std::to_string(slots) + " pages " + std::to_string(max_pps) +
                                       " layers " + c.engine.o.layers +
                                       " format " + std::to_string(desc.format_hash()) +
                                       " state " + std::to_string(desc.state_blocks()) +
                                       " blocks " + std::to_string(blocks) +
                                       " engine " + exe;
            stash->add_tier(blocks, (keep == "system" ? 1 : 2) | (whole ? 4 : 0),
                            index.empty() ? nullptr : index.c_str(), compat.c_str());
            stash->set_policy(uint32_t(i), 0, keep == "sessions" ? 1 : 0, desc.slot_bytes());
            std::fprintf(stderr, "# rank %d: qw38 s2 store %zu: %s, %llu blocks of %zu B, keep %s%s\n",
                         c.rank, i, tiers->describe(i).c_str(), (unsigned long long)blocks, desc.slot_bytes(),
                         keep.c_str(), index.empty() ? "" : ", indexed");
        }
        stash->tiers_loaded();
        stash->write_limit(uint64_t(std::llround(c.kv_wlim * 1000)));
    }

    auto sync_ple_ctx_to_dev = [&](int q) {
        if (d_ple_ctx && c.engine.has_ple)
            QW38_CK(cudaMemcpyAsync(d_ple_ctx + size_t(q) * 2, c.engine.ple_ctx[size_t(q)].data(), 8,
                                    cudaMemcpyHostToDevice, c.engine.stream));
    };
    auto sync_ple_ctx_from_dev = [&](int q) {
        if (d_ple_ctx && c.engine.has_ple)
            QW38_CK(cudaMemcpy(c.engine.ple_ctx[size_t(q)].data(), d_ple_ctx + size_t(q) * 2, 8,
                               cudaMemcpyDeviceToHost));
    };
    const auto state_pages = [&](unsigned q, unsigned pos) {
        std::vector<uint32_t> v(pos / geom::KV_PAGE);
        for (unsigned i = 0; i < v.size(); ++i) v[i] = pack_page_handle(q, i, max_pps);
        return v;
    };
    const auto drain_store = [&]() {
        if (!stash) return;
        const auto io = stash->io();
        if (!io.empty()) did_drain = true;
        tiers->run(io, c.engine.stream);
        stash->io_done();
    };

    NativeHooks h;
    h.pool.pool = &pool;
    h.pool.max_run_pages = pages / unsigned(slots);

    if (stash) {
        h.checkpoint = [&](const g6rt::ServeReq& r) -> uint32_t {
            if (!store_shared || r.prompt.empty() || r.steering.enabled || !r.images.empty()) return 0;
            const uint64_t top = std::min<uint64_t>(r.prompt.size() - 1, r.sys < 0 ? r.prompt.size() - 1 : uint64_t(r.sys));
            return uint32_t(top / geom::KV_PAGE * geom::KV_PAGE);
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
                } else hit.why = "room";
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
            require(q >= 0, "qw38 retain: run has no bound slot");
            const uint32_t p = uint32_t(std::max(0, st.valid));
            if (p > 0) {
                sync_ple_ctx_to_dev(q);
                QW38_CK(cudaStreamSynchronize(c.engine.stream));
                std::lock_guard<std::mutex> lk(ck_mu);
                StoreSlot& ss = store_slots[size_t(q)];
                if (p > ss.earned) {
                    stash->earn(p / geom::KV_PAGE - ss.earned / geom::KV_PAGE + tiers->desc().state_blocks());
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
    }

    h.bind_request = [&](unsigned slot, const g6rt::ServeReq& r) {
        require(slot < MAX_SESSIONS, "qw38 bind_request: bad slot");
        incoming_sys[slot] = r.sys;
        incoming_mrope_delta[slot] = qw38rt::qw38_mrope_delta(r.images);
        incoming_img_geom[slot].clear();
        for (const auto& im : r.images) {
            g6rt::ImageIn g;
            g.pos = im.pos; g.vit_h = im.vit_h; g.vit_w = im.vit_w; g.span = im.span;
            incoming_img_geom[slot].push_back(g);
        }
        if (stash) {
            std::lock_guard<std::mutex> lk(ck_mu);
            stash->count(r.prompt, r.sys, g6rt::steering::cache_namespace(r.steering), !claimed.empty());
            stash->io_done();
        }
    };

    h.bind = [&](const Grant& g) {
        const int q = int(g.key.slot);
        require(q >= 0 && q < slots, "qw38 bind: slot out of range");
        c.engine.reset_slot(q);
        c.engine.set_slot_mrope_delta(q, incoming_mrope_delta[size_t(q)]);
        if (g.reuse > 0 && (g.reuse & 3u) != 0) {
            int3 blk_thw;
            qw38rt::qw38_build_mrope(incoming_img_geom[size_t(q)], int(g.reuse & ~3u), 1, &blk_thw);
            c.engine.set_slot_idx_raw_thw(q, blk_thw);
        }
        QW38_CK(cudaDeviceSynchronize());
        bound[size_t(q)] = g.run;
        bound_live[size_t(q)] = true;
        if (stash) {
            StoreSlot ss{};
            ss.sys = incoming_sys[size_t(q)];
            ss.ckpt = g.req.ckpt_at > g.reuse ? g.req.ckpt_at : 0;
            if (g.reuse > 0) {
                std::lock_guard<std::mutex> lk(ck_mu);
                require(!claimed.empty(), "qw38 bind: prefix hit without claim");
                const Claimed cl = claimed.front();
                claimed.pop_front();
                const auto meta = stash->info(cl.age);
                const auto r = stash->restore(cl.tier, cl.age, state_pages(unsigned(q), g.reuse), pack_state_handle(uint32_t(q), g.reuse));
                require(r.valid == int32_t(g.reuse), "qw38 bind: restored position mismatch");
                drain_store();
                QW38_CK(cudaStreamSynchronize(c.engine.stream));
                sync_ple_ctx_from_dev(q);
                ss.hits = r.hits + 1;
                ss.pages = uint32_t(r.full_pages);
                ss.earned = g.reuse;
                ss.id = cl.age;
                ss.pf_end = meta.pf_end;
                ss.restored_end = g.reuse != uint32_t(meta.pf_end);
            }
            store_slots[size_t(q)] = ss;
        }
    };

    h.retire = [&](const Slot& s) {
        if (!stash && int(s.key.slot) >= 0 && int(s.key.slot) < slots) bound_live[s.key.slot] = false;
        QW38_CK(cudaDeviceSynchronize());
    };
    h.fence = []() { QW38_CK(cudaDeviceSynchronize()); };
    h.dispatch.bucket = [](bool, int) { return 0; };
    h.dispatch.mode = [](int, int) {};

    h.dispatch.launch = [&](const Launch& l) -> std::vector<int> {
        const s2::Step& sp = l.step;
        const int M = sp.M();
        require(M >= 1 && M <= max_rows && l.samples.size() >= size_t(M), "qw38 launch: bad M");
        std::vector<qw38::StepSeg> segs;
        std::vector<int> toks(static_cast<size_t>(M));
        std::vector<qw38rt::RowSample> sm(static_cast<size_t>(M));
        targets.assign(static_cast<size_t>(M), -1);
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const int q = sp.sess[j], r0 = sp.row0[j], n = sp.rows[j];
            segs.push_back({q, r0, sp.r[size_t(r0)].pos, n});
            for (int r = r0; r < r0 + n; ++r) {
                const s2::Row& x = sp.r[size_t(r)];
                toks[size_t(r)] = x.tok;
                targets[size_t(r)] = x.tgt;
                const Sampling& s = l.samples[size_t(r)];
                sm[size_t(r)] = qw38rt::RowSample{s.inv_temp, s.top_p, s.seed, uint32_t(x.spos)};
            }
        }
        c.engine.forward(segs, toks.data(), sm.data(), out, false);
        lp_rows = M;
        std::vector<int> ids(static_cast<size_t>(M));
        for (int r = 0; r < M; ++r) ids[size_t(r)] = out[size_t(r)].tok;
        return ids;
    };

    h.dispatch.logprob = [&](int row, bool target, int k) -> g6rt::LpRow {
        require(row >= 0 && row < lp_rows, "qw38 logprob: bad row");
        const qw38rt::RowOut& o = out[size_t(row)];
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
        require(size_t(first) + rows <= p.prompt.size() && q >= 0 && q < slots, "qw38 prefill: out of range");
        if (!p.images.empty()) c.engine.set_slot_mrope_delta(q, qw38rt::qw38_mrope_delta(p.images));
        uint32_t at = first;
        const uint32_t limit = first + rows;
        std::vector<int3> mrope_thw;
        while (at < limit) {
            uint32_t step_lim = std::min<uint32_t>(limit, at + uint32_t(max_rows));
            if (stash && p.images.empty()) {
                const uint32_t ck = store_slots[size_t(q)].ckpt;
                if (ck > at && ck < step_lim) step_lim = ck;
            }
            const int n = int(step_lim - at);
            std::vector<qw38::StepSeg> segs = {{q, 0, int(at), n}};
            if (!p.images.empty()) {
                const int vis_n = c.engine.vision.stage(p.images, int(at), n, c.engine.stream, c.rank, s.req.id);
                mrope_thw.resize(size_t(n));
                qw38rt::qw38_build_mrope(p.images, int(at), n, mrope_thw.data());
                c.engine.forward(segs, &p.prompt[at], nullptr, out, true, 0, vis_n, mrope_thw.data());
            } else {
                c.engine.forward(segs, &p.prompt[at], nullptr, out, true);
            }
            at = step_lim;
            if (stash && p.images.empty() && store_slots[size_t(q)].ckpt && at == store_slots[size_t(q)].ckpt) {
                sync_ple_ctx_to_dev(q);
                QW38_CK(cudaStreamSynchronize(c.engine.stream));
                std::lock_guard<std::mutex> lk(ck_mu);
                StoreSlot& ss = store_slots[size_t(q)];
                if (at > ss.earned) {
                    stash->earn(at / geom::KV_PAGE - ss.earned / geom::KV_PAGE + tiers->desc().state_blocks());
                    ss.earned = at;
                }
                stash->promote(p.prompt, int32_t(at), int32_t(at), int32_t(at), state_pages(unsigned(q), at),
                               pack_state_handle(uint32_t(q), at), at, ss.sys);
                drain_store();
                ss.ckpt = 0;
            }
        }
        if (stash && !p.prompt.empty() && limit == p.prompt.size() - 1 && limit > 0) {
            sync_ple_ctx_to_dev(q);
            QW38_CK(cudaStreamSynchronize(c.engine.stream));
            const g6rt::CacheNamespace ns = g6rt::steering::cache_namespace(s.req.steering);
            std::lock_guard<std::mutex> lk(ck_mu);
            StoreSlot& ss = store_slots[size_t(q)];
            if (limit > ss.earned) {
                stash->earn(limit / geom::KV_PAGE - ss.earned / geom::KV_PAGE + tiers->desc().state_blocks());
                ss.earned = limit;
            }
            ss.pf_end = int(limit);
            const bool keep_from = ss.id != 0;
            const uint64_t id = stash->capture(p.prompt, int32_t(limit), int32_t(limit), ss.pf_end,
                                               state_pages(unsigned(q), limit), pack_state_handle(uint32_t(q), limit),
                                               ss.hits, ns, ss.id, ss.pages, keep_from, 0);
            drain_store();
            if (id) {
                ss.pages = uint32_t(stash->info(id).pages);
                ss.id = id;
                ss.prompt_id = id;
            }
        }
        lp_rows = 0;
    };

    ControlLinks links = bootstrap(bc);
    try {
        if (stash && kv_index) {
            stash->keep_agreed(agree_lines(bc, links, stash->stored_summary()));
        }
        capture_barrier(bc, links);
        Service service(bc, links, std::move(h), chunk);
        service.run();
        links.shutdown();
    } catch (...) { links.shutdown(); throw; }
    if (d_ple_ctx) cudaFree(d_ple_ctx);
}

} // namespace g6s2
#endif
