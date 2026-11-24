// Qwen3.8-Flash-Next behind the S2 host service (service.h): wraps qw38rt::Engine (../qw38_step.cuh)
// with multi-session scheduling, StashPool + KvStateTiers prefix/session persistence (kv_state_desc.h qw38_kv_state),
// and cross-width live-scale state carry (GR8_KV_REWIDTH via qw38_ckpt_rewidth.h).
#ifndef G6_S2_QW38_NATIVE_CUH
#define G6_S2_QW38_NATIVE_CUH
#include "service.h"
#include "kv_state_desc.h"
#include "kv_state_mover.cuh"
#include "qw38_ckpt_rewidth.h"
#include "flash_dfl_window.h"
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
    const int K = int(sc.mtp);
    const bool use_dfl = c.engine.has_dfl && c.engine.o.dfl_k > 0;
    const int dfl_k = use_dfl ? c.engine.o.dfl_k : 0;
    const int max_k = std::max(K, dfl_k);
    const bool spec_on = (max_k > 0);
    require(sc.max_sessions == unsigned(slots) && sc.max_rows <= unsigned(max_rows),
            "qw38: --s2-sessions and --max-rows mismatch");
    require(K == 0 || (K >= 1 && K <= 7 && c.engine.has_mtp),
            "qw38: --mtp 1..7 requires loaded L048 MTP drafter");
    const unsigned max_pps = unsigned(c.engine.max_pages_per_slot());
    const unsigned pages = sc.pool_rows / geom::KV_PAGE;
    RunPool pool(pages, pages, sc.pool_rows);

    std::vector<qw38rt::RowOut> out;
    std::vector<int> targets;
    int lp_rows = 0;

    // MTP / QSpine-DF2: pending verify rows of the last trunk forward, cached K drafts, and committed end per slot
    struct Pending {
        int row0 = 0, rows = 0, pos0 = 0;
        std::array<int, 8> toks{};
        std::array<int32_t, 2> ple_ctx{geom::TOK_EOS, geom::TOK_EOS};
    };
    struct Drafts {
        std::array<int, 8> d{};
        std::array<float, 8> lp{};
        int q = -1;
        int chained_k = 0;
        int avail_k = 0;
        bool valid = false;
        bool idx_dirty = false;
        qw38rt::RowSample base_sm{};
    };
    std::array<Pending, MAX_SESSIONS> pend{};
    std::array<Drafts, MAX_SESSIONS> drafts{};
    std::array<int, MAX_SESSIONS> done_end{};
    std::array<bool, MAX_SESSIONS> slot_want_lp{};
    std::array<bool, MAX_SESSIONS> dfl_active{};
    Drafts dfl_last{};
    g6s2::FlashDflWindow dwin(slots, qw38::dfl::WIN, 0);
    int cur_solo = -1;
    std::vector<float> last_dlp;

    // KStopB early-stop lazy drafting parameters (copied up front so lambdas never read moved-from NativeHooks).
    std::array<double, 8> kstop_tau1{};
    bool kstop_early_stop = false;
    if (K > 1 && c.engine.has_mtp && c.engine.o.nr == 1 && c.engine.o.use_graph && c.engine.o.mtp_fast &&
        knob::mtp_kstopb() && int(knob::mtp_kstopb_kmin()) == 0) {
        const auto& btau = knob::mtp_kstopb_tau();
        for (size_t i = 0; i < btau.size() && i < 7; ++i) kstop_tau1[i + 1] = btau[i];
        if (const auto ts1 = knob::mtp_kstopb_tau_s(1)) {
            for (size_t i = 0; i < ts1->size() && i < 7; ++i) kstop_tau1[i + 1] = (*ts1)[i];
        }
        bool any_multi_s_override = false;
        for (int s = 2; s <= s2::S2N_SLOTS; ++s) {
            if (knob::mtp_kstopb_tau_s(uint32_t(s))) { any_multi_s_override = true; break; }
        }
        kstop_early_stop = !any_multi_s_override;
    }

    const auto commit_pending_meta = [&](int q, int kept, std::vector<int>& gdn_slots, std::vector<int>& gdn_kept) {
        if (drafts[size_t(q)].idx_dirty) {
            c.engine.mtp_restore_idx_raw(q);
            drafts[size_t(q)].idx_dirty = false;
        }
        Pending& w = pend[size_t(q)];
        if (w.rows <= 0) return;
        kept = std::max(0, std::min(w.rows, kept));
        if (kept < w.rows) {
            c.engine.mtp_rollback_slot(q, kept);
            if (kept >= 2) {
                c.engine.ple_ctx[size_t(q)][0] = w.toks[size_t(kept - 2)];
                c.engine.ple_ctx[size_t(q)][1] = w.toks[size_t(kept - 1)];
            } else if (kept == 1) {
                c.engine.ple_ctx[size_t(q)][0] = w.ple_ctx[1];
                c.engine.ple_ctx[size_t(q)][1] = w.toks[0];
            } else {
                c.engine.ple_ctx[size_t(q)] = w.ple_ctx;
            }
        }
        if (kept > 0) {
            gdn_slots.push_back(q);
            gdn_kept.push_back(kept);
        }
        if (kept > 1) {
            c.engine.mtp_sync_kv(q, w.row0, w.pos0, kept - 1, &w.toks[1]);
        }
        if (kept > 0) {
            c.engine.mtp_save_last_hyper(q, w.row0 + kept - 1);
        }
        done_end[size_t(q)] = w.pos0 + kept;
        w.rows = 0;
    };

    const auto commit_pending = [&](int q, int kept) {
        std::vector<int> gs, gk;
        commit_pending_meta(q, kept, gs, gk);
        if (!gs.empty()) c.engine.mtp_commit_gdn_batch(gs, gk);
    };

    const auto settle = [&](const s2::Step* sp) {
        if (!spec_on) return;
        std::vector<int> gs, gk;
        for (int q = 0; q < slots; ++q) {
            Pending& w = pend[size_t(q)];
            if (w.rows <= 0) continue;
            int kept = 0;
            for (size_t j = 0; sp && j < sp->sess.size(); ++j)
                if (sp->sess[j] == q) kept = std::max(0, std::min(w.rows, sp->r[size_t(sp->row0[j])].pos - w.pos0));
            commit_pending_meta(q, kept, gs, gk);
        }
        if (!gs.empty()) c.engine.mtp_commit_gdn_batch(gs, gk);
    };

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
                    reinterpret_cast<uint8_t*>(rl.kv.idx_raw),
                });
            } else {
                gl.push_back(Qw38GdnStateLayer{
                    reinterpret_cast<uint8_t*>(rl.gdn_st.S),
                    reinterpret_cast<uint8_t*>(rl.gdn_st.conv),
                });
            }
        }
        if (K > 0 && c.engine.has_mtp) {
            const auto& mkv = c.engine.mtp_w.kv[0];
            fl.push_back(Qw38FullKvLayer{
                reinterpret_cast<uint8_t*>(mkv.kv_pages),
                reinterpret_cast<uint8_t*>(mkv.kv_staging),
                reinterpret_cast<uint8_t*>(mkv.idx_pages),
                reinterpret_cast<uint8_t*>(mkv.idx_staging),
                reinterpret_cast<uint8_t*>(mkv.idx_raw),
            });
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
        if (specs.empty()) {
            // Size the default in-memory prefix cache to hold a full-cap conversation plus sibling state (clamped 512M..4096M).
            const uint64_t want_blocks = uint64_t(max_pps) + 2ULL * desc.state_blocks() + 16ULL;
            const uint64_t want_mib = (want_blocks * uint64_t(desc.slot_bytes()) + ((1ULL << 20) - 1ULL)) >> 20;
            const uint64_t def_mib = std::max<uint64_t>(512ULL, std::min<uint64_t>(4096ULL, want_mib));
            specs.push_back("mem:,size=" + std::to_string(def_mib) + "M,keep=" + c.kv_keep);
        }
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
                                       " mtp " + std::to_string(K > 0 ? 1 : 0) +
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
            // In Strata mode, splitting a prompt at a mid-prompt 64-token page boundary forces a second full pass
            // over cold MoE experts across PCIe (+50..220 ms TTFT). stash->capture() at end-of-prefill already
            // caches the full prompt prefix without splitting!
            if (c.engine.o.strata.enabled) return 0;
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
            if (K > 0 && drafts[size_t(q)].idx_dirty) {
                c.engine.mtp_restore_idx_raw(q);
                drafts[size_t(q)].idx_dirty = false;
            }
            if (spec_on && pend[size_t(q)].rows > 0) {
                commit_pending(q, int(p) - pend[size_t(q)].pos0);
            }
            if (p > 0) {
                sync_ple_ctx_to_dev(q);
                c.engine.stage_partial_page(q, p, true);
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
        slot_want_lp[slot] = (r.lp >= 0 || r.plp >= 0);
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
        pend[size_t(q)] = Pending{};
        drafts[size_t(q)] = Drafts{};
        if (use_dfl) {
            dwin.bind(q, g.req.prompt_tokens, g.reuse);
            dfl_active[size_t(q)] = false;
        }
        done_end[size_t(q)] = int(g.reuse);
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
                c.engine.stage_partial_page(q, g.reuse, false);
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
        if (int(s.key.slot) >= 0 && int(s.key.slot) < slots) {
            if (drafts[s.key.slot].idx_dirty) c.engine.mtp_restore_idx_raw(int(s.key.slot));
            pend[s.key.slot] = Pending{};
            drafts[s.key.slot] = Drafts{};
            slot_want_lp[s.key.slot] = false;
            dfl_active[s.key.slot] = false;
            if (cur_solo == int(s.key.slot)) dfl_last = Drafts{};
        }
        QW38_CK(cudaDeviceSynchronize());
    };
    h.fence = []() { QW38_CK(cudaDeviceSynchronize()); };
    h.dispatch.bucket = [](bool, int) { return 0; };
    h.dispatch.mode = [&](int, int next_solo) {
        cur_solo = next_solo;
        if (next_solo < 0) {
            dfl_active.fill(false);
            dfl_last = Drafts{};
        }
    };

    if (use_dfl) {
        h.dfl_kv = dfl_k;
        if (c.engine.o.dfl_dp >= 2) {
            h.dfl_spec.on = true;
            h.dfl_spec.levels.clear();
            for (int k = 1; k <= dfl_k; ++k) h.dfl_spec.levels.push_back(k);
            h.dfl_spec.draft_round = 3.45;
            static const double t1[9] = {0, 27.1, 29.9, 32.7, 35.5, 38.2, 40.9, 43.6, 46.3};
            static const double t2[9] = {0, 17.5, 19.3, 21.1, 22.9, 24.7, 26.5, 28.3, 30.1};
            const double* tt = (c.engine.o.nr == 1) ? t1 : t2;
            for (int r = 0; r <= g6rt::SpecLen::KX + 1; ++r) h.dfl_spec.table[r] = tt[r];
            // Pre-seed calibrated probability buckets so SpecLen::choose trusts Viterbi calibrated lp on round 1
            static const double bmid[g6rt::SpecLen::NB] = {0.08, 0.30, 0.50, 0.675, 0.815, 0.915, 0.97, 0.995};
            h.dfl_spec.prior = 2.0;
            for (int j = 1; j <= g6rt::SpecLen::KX; ++j) {
                for (int b = 0; b < g6rt::SpecLen::NB; ++b) {
                    h.dfl_spec.bn[j][b] = 12.0;
                    h.dfl_spec.bk[j][b] = 12.0 * bmid[b];
                }
            }
        }
        h.dispatch.dfl_ok = [&](int q) { return dwin.ok(q); };
        h.dispatch.dfl_start = [&](int q) -> int {
            dfl_active[size_t(q)] = true;
            if (dwin.fills(q)) {
                c.engine.dfl_finish_prompt_range(dwin.lo, dwin.from, dwin.hi);
            }
            return dwin.start(q);
        };
        h.dispatch.dfl = [&](const s2::DflReq& r, const Sampling&) -> s2::DflDrafts {
            if (r.rows > 0) {
                c.engine.dfl.ingest_context(c.engine.dfl_step_taps, r.rows, r.pos0, c.engine.stream);
            }
            s2::DflDrafts o{};
            if (!r.query) return o;
            c.engine.dfl.ctx_lo = r.lo;
            qw38::dfl::DraftResult dr{};
            const int dp_m = (c.engine.o.dfl_dp >= 1) ? 1 : 0;
            c.engine.dfl_draft_step(r.tok0, r.pos0 + r.rows, dfl_k, dr, dp_m);
            o.tok[0] = r.tok0;
            o.lp[0] = 0.0f;
            dfl_last = Drafts{};
            dfl_last.q = r.pos0 + r.rows;
            dfl_last.chained_k = dr.K;
            dfl_last.valid = (dr.K > 0);
            for (int i = 0; i < dr.K && i + 1 < s2::DFL_B; ++i) {
                o.tok[i + 1] = dr.tok[i];
                o.lp[i + 1] = dr.lp[i];
                if (i < 8) dfl_last.d[size_t(i)] = dr.tok[i];
            }
            return o;
        };
        if (c.rank == 0) {
            std::fprintf(stderr, "# qw38 s2: QSpine-DF2 enabled for solo c1 (k<=%d, dp=%d, adaptive=%d, hybrid_mtp=%d)\n",
                         dfl_k, c.engine.o.dfl_dp, h.dfl_spec.on ? 1 : 0, (K > 0 && c.engine.has_mtp) ? K : 0);
        }
    }

    h.dispatch.launch = [&](const Launch& l) -> std::vector<int> {
        const s2::Step& sp = l.step;
        const int M = sp.M();
        require(M >= 1 && M <= max_rows && l.samples.size() >= size_t(M), "qw38 launch: bad M");
        if (sp.kind == s2::Step::CHAIN) {
            std::vector<int> u(size_t(M), 0);
            last_dlp.assign(size_t(M), 0.0f);
            for (size_t j = 0; j < sp.sess.size(); ++j) {
                const int q = sp.sess[j];
                Drafts& d = drafts[size_t(q)];
                const s2::Row& x = sp.r[size_t(sp.row0[j])];
                if (!d.valid && use_dfl) {
                    // Single demotion transition round from solo QSpine-DF2 -> multi-session L48 MTP (100% lossless)
                    u[size_t(sp.row0[j])] = x.tok;
                    continue;
                }
                const int i = x.pos - d.q;
                require(K > 0 && sp.rows[j] == 1 && d.valid && i >= 1 && i < K && x.tok == d.d[size_t(i - 1)],
                        "qw38 chain step does not continue cached draft");
                if (i >= d.avail_k) {
                    if (kstop_early_stop && sp.sess.size() == 1) {
                        double pcum = 1.0;
                        bool can_pass = true;
                        for (int t = 0; t < i; ++t) {
                            pcum *= std::exp(double(d.lp[size_t(t)]));
                            if (pcum < kstop_tau1[size_t(t + 1)]) { can_pass = false; break; }
                        }
                        if (can_pass && pcum < kstop_tau1[size_t(i + 1)]) can_pass = false;
                        if (!can_pass) {
                            for (int t = i; t < K; ++t) {
                                d.d[size_t(t)] = 0;
                                d.lp[size_t(t)] = -100.0f;
                            }
                            d.avail_k = K;
                        } else {
                            const bool last_k = (i + 1 == K);
                            std::vector<int> qv{q}, pv{d.q + i}, tv{d.d[size_t(i - 1)]}, dout;
                            std::vector<qw38rt::RowSample> smv{d.base_sm};
                            std::vector<float> dlp;
                            c.engine.mtp_draft_k(qv, pv, tv, smv, 1, dout, &dlp, last_k ? 2 : 3);
                            require(dout.size() == 1, "qw38 early-stop lazy draft: bad output size");
                            d.d[size_t(i)] = dout[0];
                            d.lp[size_t(i)] = dlp[0];
                            d.avail_k = i + 1;
                            d.idx_dirty = !last_k;
                        }
                    } else {
                        const int rem = K - d.avail_k;
                        std::vector<int> qv{q}, pv{d.q + d.avail_k}, tv{d.d[size_t(d.avail_k - 1)]}, dout;
                        std::vector<qw38rt::RowSample> smv{d.base_sm};
                        std::vector<float> dlp;
                        c.engine.mtp_draft_k(qv, pv, tv, smv, rem, dout, &dlp, 2);
                        require(dout.size() == size_t(rem), "qw38 lazy draft cont: bad output size");
                        for (int t = 0; t < rem; ++t) {
                            d.d[size_t(d.avail_k + t)] = dout[size_t(t)];
                            d.lp[size_t(d.avail_k + t)] = dlp[size_t(t)];
                        }
                        d.avail_k = K;
                        d.idx_dirty = false;
                    }
                }
                d.chained_k = std::max(d.chained_k, i + 1);
                u[size_t(sp.row0[j])] = d.d[size_t(i)];
                last_dlp[size_t(sp.row0[j])] = d.lp[size_t(i)];
            }
            return u;
        }
        if (sp.kind == s2::Step::DRAFT) {
            require(spec_on, "qw38 draft step at MTP 0 without QSpine-DF2");
            std::vector<int> u(size_t(M), 0), ds, dq, dg, dlast, gs, gk;
            std::vector<qw38rt::RowSample> dsm;
            const bool solo_dfl = use_dfl && (sp.sess.size() == 1) && dfl_active[size_t(sp.sess[0])];
            last_dlp.assign(size_t(M), 0.0f);
            for (size_t j = 0; j < sp.sess.size(); ++j) {
                const int q = sp.sess[j], r0 = sp.row0[j], n = sp.rows[j];
                if (drafts[size_t(q)].idx_dirty) {
                    c.engine.mtp_restore_idx_raw(q);
                    drafts[size_t(q)].idx_dirty = false;
                }
                for (int i = 1; i < n; ++i)
                    require(sp.r[size_t(r0 + i)].pos == sp.r[size_t(r0)].pos + i, "qw38 draft rows are not consecutive");
                Pending& w = pend[size_t(q)];
                if (w.rows > 0) {
                    require(sp.r[size_t(r0)].pos == w.pos0 && n <= w.rows, "qw38 draft rows mismatch pending verify");
                    commit_pending_meta(q, n, gs, gk);
                }
                const s2::Row& last = sp.r[size_t(r0 + n - 1)];
                require(last.pos < done_end[size_t(q)], "qw38 draft from uncommitted row");
                const int ev = j < sp.ev.size() ? sp.ev[j] : int(sp.event);
                if (sp.round < 0 && ev != s2::Step::TAIL) continue;
                if (solo_dfl || K <= 0 || !c.engine.has_mtp) {
                    // Solo QSpine-DF2 round: GDN commit already queued; skip L48 draft compute (0.00 ms overhead!)
                    drafts[size_t(q)] = Drafts{};
                    u[size_t(r0 + n - 1)] = last.tok;
                    continue;
                }
                const Sampling& sa = l.samples[size_t(r0 + n - 1)];
                ds.push_back(q);
                dq.push_back(last.pos);
                dg.push_back(last.tok);
                dlast.push_back(r0 + n - 1);
                dsm.push_back(qw38rt::RowSample{sa.inv_temp, sa.top_p, sa.seed, 0u});
            }
            if (!gs.empty()) c.engine.mtp_commit_gdn_batch(gs, gk);
            if (!ds.empty()) {
                const int k1 = (kstop_early_stop && ds.size() == 1)
                    ? 1
                    : ((K > 3 && ds.size() == 1 && c.engine.o.nr == 1 && c.engine.o.use_graph && c.engine.o.mtp_fast) ? 3 : K);
                const int dmode = (k1 < K) ? 1 : 0;
                std::vector<int> dout;
                std::vector<float> dlp;
                c.engine.mtp_draft_k(ds, dq, dg, dsm, k1, dout, &dlp, dmode);
                require(dout.size() == ds.size() * size_t(k1), "qw38 draft: bad output size");
                for (size_t i = 0; i < ds.size(); ++i) {
                    Drafts& d = drafts[size_t(ds[i])];
                    d = Drafts{};
                    for (int t = 0; t < k1; ++t) {
                        d.d[size_t(t)] = dout[i * size_t(k1) + size_t(t)];
                        d.lp[size_t(t)] = dlp[i * size_t(k1) + size_t(t)];
                    }
                    d.q = dq[i];
                    d.chained_k = 1;
                    d.avail_k = k1;
                    d.valid = true;
                    d.idx_dirty = (dmode == 1);
                    d.base_sm = dsm[i];
                    u[size_t(dlast[i])] = d.d[0];
                    last_dlp[size_t(dlast[i])] = d.lp[0];
                }
            }
            return u;
        }
        if (spec_on) settle(&sp);
        std::vector<qw38::StepSeg> segs;
        std::vector<int> toks(static_cast<size_t>(M));
        std::vector<qw38rt::RowSample> sm(static_cast<size_t>(M));
        targets.assign(static_cast<size_t>(M), -1);
        std::vector<bool> is_verify(sp.sess.size(), false);
        int verify_k = 0;
        bool exact_head = false;
        for (size_t j = 0; j < sp.sess.size(); ++j) {
            const int q = sp.sess[j], r0 = sp.row0[j], n = sp.rows[j];
            const int pos0 = sp.r[size_t(r0)].pos;
            if (slot_want_lp[size_t(q)]) exact_head = true;
            // Distinguish forced-insert MAIN chunks (single session from planner::forced, e.g. forced tool_choice)
            // from MTP/QSpine-DF2 verify steps (round_all). Single-session round_all never uses waterfill, so its
            // verify rows always match the cached draft tokens and length.
            bool ver = (spec_on && sp.round >= 0 && n > 1);
            if (ver && sp.sess.size() == 1) {
                if (dfl_active[size_t(q)]) {
                    if (n > dfl_k + 1 || !dfl_last.valid || pos0 != dfl_last.q || (n - 1) > dfl_last.chained_k) {
                        ver = false;
                    } else {
                        for (int i = 1; i < n; ++i) {
                            if (sp.r[size_t(r0 + i)].tok != dfl_last.d[size_t(i - 1)]) { ver = false; break; }
                        }
                    }
                } else {
                    const Drafts& d = drafts[size_t(q)];
                    if (n > K + 1 || !d.valid || pos0 != d.q + 1 || (n - 1) > d.chained_k) {
                        ver = false;
                    } else {
                        for (int i = 1; i < n; ++i) {
                            if (sp.r[size_t(r0 + i)].tok != d.d[size_t(i - 1)]) { ver = false; break; }
                        }
                    }
                }
            }
            if (ver) {
                is_verify[j] = true;
                verify_k = std::max(verify_k, n - 1);
            }
            segs.push_back({q, r0, pos0, n});
            for (int r = r0; r < r0 + n; ++r) {
                const s2::Row& x = sp.r[size_t(r)];
                toks[size_t(r)] = x.tok;
                targets[size_t(r)] = x.tgt;
                const Sampling& s = l.samples[size_t(r)];
                sm[size_t(r)] = qw38rt::RowSample{s.inv_temp, s.top_p, s.seed, uint32_t(x.spos)};
            }
        }
        const auto saved_ple_ctx = (verify_k > 0) ? c.engine.ple_ctx : std::vector<std::array<int32_t, 2>>{};
        c.engine.forward(segs, toks.data(), sm.data(), out, false, verify_k, 0, nullptr, false, exact_head);
        if (spec_on) {
            for (size_t j = 0; j < sp.sess.size(); ++j) {
                const int q = sp.sess[j], r0 = sp.row0[j], n = sp.rows[j];
                const int pos0 = sp.r[size_t(r0)].pos;
                if (!is_verify[j]) {
                    if (drafts[size_t(q)].idx_dirty) {
                        c.engine.mtp_restore_idx_raw(q);
                        drafts[size_t(q)].idx_dirty = false;
                    }
                    if (n > 1) c.engine.mtp_sync_kv(q, r0, pos0, n - 1, &toks[size_t(r0 + 1)]);
                    c.engine.mtp_save_last_hyper(q, r0 + n - 1);
                    done_end[size_t(q)] = pos0 + n;
                } else {
                    require(n <= max_k + 1, "qw38 verify: n > max_k + 1");
                    Pending w{};
                    w.row0 = r0;
                    w.rows = n;
                    w.pos0 = pos0;
                    for (int i = 0; i < n; ++i) w.toks[size_t(i)] = toks[size_t(r0 + i)];
                    w.ple_ctx = saved_ple_ctx[size_t(q)];
                    pend[size_t(q)] = w;
                }
            }
        }
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
            if (o.top[i].id == id) { e.lp = o.top[i].logit - o.lse; e.rank = i < size_t(g6rt::LP_MAX) ? int(i) + 1 : -1; break; }
        return e;
    };

    h.dispatch.prefill = [&](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) {
        const int q = int(s.key.slot);
        require(size_t(first) + rows <= p.prompt.size() && q >= 0 && q < slots, "qw38 prefill: out of range");
        if (!p.images.empty()) c.engine.set_slot_mrope_delta(q, qw38rt::qw38_mrope_delta(p.images));
        if (use_dfl) {
            if (dwin.quantum(q, s.req.id, uint32_t(p.prompt.size()), first) || dwin.owner == q) {
                c.engine.dfl_begin_prompt_window(dwin.lo, dwin.hi);
            }
        }
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
            // Do not discard other live sessions' uncommitted verify rows during prefill on slot q.
            if (K > 0 && drafts[size_t(q)].idx_dirty) {
                c.engine.mtp_restore_idx_raw(q);
                drafts[size_t(q)].idx_dirty = false;
            }
            if (spec_on && pend[size_t(q)].rows > 0) commit_pending(q, 0);
            std::vector<qw38::StepSeg> segs = {{q, 0, int(at), n}};
            if (!p.images.empty()) {
                const int vis_n = c.engine.vision.stage(p.images, int(at), n, c.engine.stream, c.rank, s.req.id);
                mrope_thw.resize(size_t(n));
                qw38rt::qw38_build_mrope(p.images, int(at), n, mrope_thw.data());
                c.engine.forward(segs, &p.prompt[at], nullptr, out, true, 0, vis_n, mrope_thw.data());
            } else {
                c.engine.forward(segs, &p.prompt[at], nullptr, out, true);
            }
            if (spec_on) {
                if (K > 0 && c.engine.has_mtp) {
                    const int sync_n = std::min(n, int(p.prompt.size()) - 1 - int(at));
                    if (sync_n > 0) c.engine.mtp_sync_kv(q, 0, int(at), sync_n, &p.prompt[at + 1]);
                }
                done_end[size_t(q)] = int(at) + n;
            }
            at = step_lim;
            if (at < limit && p.images.empty()) {
                uint32_t next_lim = std::min<uint32_t>(limit, at + uint32_t(max_rows));
                if (stash) {
                    const uint32_t ck = store_slots[size_t(q)].ckpt;
                    if (ck > at && ck < next_lim) next_lim = ck;
                }
                c.engine.ple_prefetch_next(q, int(at), &p.prompt[at], int(next_lim - at));
            }
            if (stash && p.images.empty() && store_slots[size_t(q)].ckpt && at == store_slots[size_t(q)].ckpt) {
                sync_ple_ctx_to_dev(q);
                c.engine.stage_partial_page(q, at, true);
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
        if (use_dfl) c.engine.dfl_end_prompt_capture();
        if (stash && !p.prompt.empty() && limit == p.prompt.size() - 1 && limit > 0) {
            sync_ple_ctx_to_dev(q);
            c.engine.stage_partial_page(q, limit, true);
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

    if (K > 0) {
        h.dispatch.draft_lp = [&](int row) -> float {
            return (row >= 0 && row < int(last_dlp.size())) ? last_dlp[size_t(row)] : 0.0f;
        };
        const int lplog = int(knob::mtp_lplog());
        h.kstop.log = (lplog == 2 || (lplog == 1 && c.rank == 0));
        if (knob::mtp_kstopb()) {
            h.kstop.b.mode = 1;
            h.kstop.b.kmin = std::max(0, std::min(K, int(knob::mtp_kstopb_kmin())));
            const auto& btau = knob::mtp_kstopb_tau();
            for (int s = 1; s <= s2::S2N_SLOTS; ++s) {
                for (size_t i = 0; i < btau.size() && i < 7; ++i) h.kstop.b.tau[s][i + 1] = btau[i];
                if (const auto ts = knob::mtp_kstopb_tau_s(uint32_t(s))) {
                    for (size_t i = 0; i < ts->size() && i < 7; ++i) h.kstop.b.tau[s][i + 1] = (*ts)[i];
                }
            }
            if (c.rank == 0) {
                std::fprintf(stderr, "# qw38 s2 MTP KStopB: ON (kmin=%d, K=%d, early_stop=%d, tau=%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)\n",
                             h.kstop.b.kmin, K, int(kstop_early_stop),
                             h.kstop.b.tau[1][1], h.kstop.b.tau[1][2], h.kstop.b.tau[1][3],
                             h.kstop.b.tau[1][4], h.kstop.b.tau[1][5], h.kstop.b.tau[1][6], h.kstop.b.tau[1][7]);
            }
        }
    }

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
