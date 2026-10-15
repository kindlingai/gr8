// Native dispatch half of Service. Requires fc2's DEC_S2 rank.cuh APIs (cdddf65+).
// Opt-in integration only: this header is NOT included by legacy g6m4 today.
// Real variable-run bind/prefill view selection remains TP4/fc2-owned.
#pragma once
#ifdef DEC_S2
#include "service.h"
#include "../prefill.cuh"
#include "../../comms/fabric.h"
#include <chrono>
#include <map>
#include <set>

namespace g6s2 {
// wall clock in ms since the Unix epoch: the ranks' logs carry no timestamps, and the skew between ranks at a bind or a
// prefill start is what a silent GPU wait timeout (G6_SPIN_TIMEOUT_CYC) turns into an error word
inline long long s2_wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
// Use g6m4's EXISTING capture arrays/main_enqueue. In particular solo_main must
// replay its production split3 A/B graphs, not capture an S2 replacement graph.
struct NativeContext {
    g6rt::Model& main;
    g6rt::Draft& draft;
    g6rt::Prefill& prefill;
    g6rt::Prefill& prefill2;
    g6rt::DraftPrefill& draft_prefill;
    g6::Fabric& main_fabric;
    g6::Fabric& draft_fabric;
    g6::Fabric& prefill_fabric;
    g6::Fabric& prefill_fabric2;
    cudaStream_t stream;
    g6rt::PfLanes* lane_streams = nullptr;
    g6rt::StepIn *input, *draft_input;
    g6rt::StepOut *output, *draft_output;
    int prefill_rows = 0, split = 3;
    bool two_lanes = true;
    std::function<void(const g6rt::StepIn&)> enqueue_main; // main fabric (head payload from the rows' top-list flags) + CP collectives past R
    std::function<void(int bucket, int rows)> solo_main;
    std::function<void(int bucket, bool chain, int rows)> solo_draft;
    // Must select the canonical run in BOTH main and draft prefill bindings,
    // including QO offsets. begin/end cannot reset KV or other sessions.
    std::function<void(const Slot&)> begin_prefill;
    std::function<void()> end_prefill;
    // optional: around the main prefill step only, as legacy g6m4 (G6_CP staged view, cp_pf_enter / cp_pf_exit)
    std::function<void()> pf_enter, pf_exit;
    // optional: runtime error words the adapter does not own (CP fabric / CP staging proxy); nonzero fails the epoch
    std::function<unsigned()> extra_error;
};
class NativeDispatch {
    NativeContext c_;
    // first-step timer: off-by-default per-step timer (design r3/r4); set from G6_STEP_TMR=1 at construction
    const bool tmr_ = ::getenv("G6_STEP_TMR") && *::getenv("G6_STEP_TMR") == '1';
    // G6_S2_ROUNDLOG=1 (rank 0; g6m4 allocates the main model's route copies): one "# s2_round" line per decode step
    const bool rlog_ = c_.main.rank == 0 && c_.main.rlog && ::getenv("G6_S2_ROUNDLOG") && *::getenv("G6_S2_ROUNDLOG") == '1';
    static const char* step_kind(const Launch& l) {
        return l.step.kind == s2::Step::MAIN ? "main" : (l.step.kind == s2::Step::CHAIN ? "chain" : "draft");
    }
    // first-step timer: E1/E2 event label (design r3/r4).
    // first-step timer r3: the header is derived from the events of the
    // sessions ACTUALLY in the step, not the Step::event default: a CHAIN step
    // never set s.event (printed ROUND_MAIN while its sessions said TAIL/DECODE1),
    // and a mixed DRAFT step inherited v.event before filtering. Uniform -> that
    // class; mixed -> MIXED_MAIN; no per-session events (hand-built test steps)
    // -> the step-level field. The per-session `reqs slot=req:event` stays
    // authoritative either way.
    static const char* step_event(const Launch& l) {
        int cls = -1;
        for (size_t j = 0; j < l.events.size(); ++j) {
            const int e = l.events[j];
            if (cls == -1) cls = e;
            else if (cls != e) return "MIXED_MAIN";
        }
        if (cls == -1) cls = l.event;
        return cls == s2::Step::TAIL ? "TAIL_MAIN" : cls == s2::Step::DECODE1 ? "DECODE_MAIN_1" : "ROUND_MAIN";
    }
    g6rt::S2In *layout_ = nullptr, *draft_layout_ = nullptr;
    const g6rt::S2In *previous_layout_ = nullptr, *previous_draft_layout_ = nullptr;
    uint64_t prefill_sequence_ = 1, prefill_sequence2_ = 1;
    g6rt::LpStep lp_{};   // logprobs of the last main step, fetched when a session in it asked (lp_rows_ rows; 0 = none)
    int lp_rows_ = 0;
    struct Graphs { cudaGraphExec_t a = nullptr, b = nullptr; };
    std::map<std::string, Graphs> graphs_;
    static void destroy(Graphs& g) {
        if (g.a) G6RT_CK(cudaGraphExecDestroy(g.a));
        if (g.b) G6RT_CK(cudaGraphExecDestroy(g.b));
        g.a = g.b = nullptr;
    }
    cudaGraphExec_t capture(const std::function<void()>& body) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        G6RT_CK(cudaStreamBeginCapture(c_.stream,cudaStreamCaptureModeThreadLocal));
        body(); G6RT_CK(cudaStreamEndCapture(c_.stream,&graph));
        G6RT_CK(cudaGraphInstantiate(&exec,graph,0)); G6RT_CK(cudaGraphDestroy(graph)); return exec;
    }
    void fence() { G6RT_CK(cudaStreamSynchronize(c_.stream)); }
    // first-step timer: "slot=req:event" per session, in step order.
    // event code: 0 = TAIL (E1), 1 = DECODE1 (E2), 2 = ROUND.
    static std::string req_list(const Launch& l) {
        std::string s;
        for (size_t j = 0; j < l.requests.size(); ++j) {
            if (j) s += ",";
            const int e = l.events.empty() ? int(s2::Step::ROUND) : l.events[j];
            s += std::to_string(l.req_slots[j]) + "=" + std::to_string(l.requests[j]) + ":" + std::to_string(e);
        }
        return s;
    }
    void mode(int old, int next) {
        const auto t0 = std::chrono::steady_clock::now(); fence();
        if (old >= 0) g6rt::s2_hprev_from_solo(c_.draft,old,c_.stream);
        if (next >= 0) {
            // the survivor's own pages (set by bind)
            const std::vector<int>& run = c_.main.s2->run[next];
            const bool ex = c_.main.s2->exclusive[next];
            require(!run.empty() && run == c_.draft.d.s2->run[next] && ex == c_.draft.d.s2->exclusive[next]
                     && c_.main.s2->ex_split[next] == c_.draft.d.s2->ex_split[next], "main/draft survivor runs differ");
            if (ex)
            {
                // a run past R: on the pool's page classes its own list -> its CP map; the identity run -> the
                // tables as built. ex_split records which at bind (the pool's mode), not the list's shape
                if (c_.main.s2->ex_split[next])
                {
                    const int most = std::max(c_.main.s2->pool_pages, c_.draft.d.s2->pool_pages);
                    g6rt::s2_solo_cp_pages(c_.main, run, most);
                    g6rt::s2_solo_cp_pages(c_.draft.d, run, most);
                }
                else { g6rt::s2_solo_built(c_.main); g6rt::s2_solo_built(c_.draft.d); }
            }
            else { g6rt::s2_solo_pages(c_.main,run); g6rt::s2_solo_pages(c_.draft.d,run); }
            g6rt::s2_hprev_to_solo(c_.draft,next,c_.stream);
        }
        c_.main.s2on = c_.draft.d.s2on = next < 0; fence();
        fprintf(stderr,"# s2 mode rank %d solo %d -> %d: %.3f ms (tables + hprev; no KV relocation)\n",c_.main.rank,old,next,
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
    }
    // G6_S2_ROUNDLOG: kind, round id and active sessions of the whole round (-1 / 0 outside a round), sessions in this step
    // (a round is one main step, one draft step and the chain steps, each over all its sessions), rows, launch+fence ms (as G6_STEP_TMR), path, and for main steps the
    // distinct routed experts per MoE layer (mean, sum, list; the G6_ROUTE_LOG data)
    void round_line(const Launch& l, int rows, std::chrono::steady_clock::time_point t0) {
        const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        std::string ex;
        if (l.step.kind == s2::Step::MAIN) {
            std::vector<int> n; g6rt::route_counts(c_.main,n);
            long sum = 0; std::string list;
            for (size_t i = 0; i < n.size(); ++i) { sum += n[i]; list += (i ? "," : "") + std::to_string(n[i]); }
            char b[64]; snprintf(b,sizeof b," ex_mean %.3f ex_sum %ld ex ",n.empty() ? 0.0 : double(sum)/n.size(),sum);
            ex = b + list;
        }
        fprintf(stderr,"# s2_round step %llu %s round %ld active %d S %zu M %d ms %.3f path %s%s\n",(unsigned long long)l.sequence,
            step_kind(l),l.step.round,l.step.round_active,l.step.sess.size(),rows,ms,l.solo >= 0 ? "solo" : "reuse",ex.c_str());
    }
    std::vector<int> launch(const Launch& l) {
        const bool draft = l.step.kind != s2::Step::MAIN, chain = l.step.kind == s2::Step::CHAIN;
        auto& model = draft ? c_.draft.d : c_.main;
        auto& fabric = draft ? c_.draft_fabric : c_.main_fabric;
        const int rows = l.step.M();
        g6rt::StepIn input{}; g6rt::S2In layout{}; native_inputs(l,input,layout);
        if (l.solo >= 0) input.page_bound = model.s2->pool_pages;   // the solo tables hold -1 past the session's pages
        if (draft && c_.main.rs) for (int r = 0; r < rows; ++r) input.srow[r].inv_temp = 0.f;   // rejection sampling: greedy drafts
        require(model.s2on == (l.solo < 0), "native solo mode differs from host");
        model.s2->h = layout;
        model.s2->bucket = l.bucket;
        for (int j = 0; j < g6rt::S2N; ++j) model.s2->bk[j] = l.bucket;   // until the attention reads its contexts from the cells (S2_ATTN_LAYOUT_FREE)
        // every reachable key is captured at boot (capture_all); an S2 launch never captures
        if (l.solo < 0 && !graphs_.count(l.graph_key))
            throw std::runtime_error("S2 launch of an uncaptured graph key " + l.graph_key + " (" + step_kind(l) + " M " + std::to_string(rows) +
                 " bucket " + std::to_string(l.bucket) + "): not in the boot capture set");
        *(draft ? c_.draft_input : c_.input) = input;
        *(draft ? draft_layout_ : layout_) = layout;
        if (draft) {
            std::vector<uint32_t> sizes(3,uint32_t(rows*::HID*2));
            sizes[2] = uint32_t(glue::head_bytes(rows,model.nc)); c_.draft_fabric.enqueue_step(input.step,sizes);
        } else c_.enqueue_main(input); // preserves existing RS order/fabric layout
        const auto tl0 = tmr_ || rlog_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}; // after fabric enqueue, BEFORE the solo/S2 launch branches (design r4 fix 2)
        if (l.solo >= 0) {
            if (draft) c_.solo_draft(l.buckets[0],chain,rows); else c_.solo_main(l.buckets[0],rows);
        } else {
            const auto& g = graphs_.at(l.graph_key); G6RT_CK(cudaGraphLaunch(g.a,c_.stream));
            if (g.b) G6RT_CK(cudaGraphLaunch(g.b,c_.stream));
        }
        fence(); const auto out = *(draft ? c_.draft_output : c_.output);
        if (!draft) {
            g6rt::evprof_read(rows,c_.stream,c_.main.rank);
            lp_rows_ = 0;
            if (std::any_of(l.samples.begin(),l.samples.end(),[](const Sampling& s){return s.lp >= 0 || s.plp >= 0;})) {
                g6rt::lp_fetch(c_.main,out,input,lp_); lp_rows_ = rows;
            }
        }
        if (tmr_) // first-step timer: launch+fence span (design r4); labels per r3, from the PRE-insertion decision, never graphs_.count
            fprintf(stderr,"# step_tmr rank %d step %llu %s %s %s %lld us (graph_key %s rows %d ctx %d reqs %s)\n",c_.main.rank,
                (unsigned long long)input.step,step_kind(l),step_event(l),
                l.solo >= 0 ? "SOLO_PRECAPTURED" : "S2_REUSE",
                (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tl0).count(),
                l.graph_key.c_str(),rows,int(l.context),req_list(l).c_str());
        if (rlog_) round_line(l, rows, tl0);
        const unsigned fe = *fabric.dev().error, ae = out.attn_err, ce = c_.extra_error ? c_.extra_error() : 0u;
        if (fe || ae || ce) // S2 error words: name the error word before the fail-closed require
            fprintf(stderr,"# S2 step error words: fabric %u attn %u cp %u (%s step %llu rows %d solo %d ctx %d)\n",fe,ae,ce,
                draft ? (chain ? "chain" : "draft") : "main",(unsigned long long)input.step,rows,l.solo,int(l.context));
        require(!fe && !ae && !ce, "S2 native step fabric/attention error");
        return {out.tok,out.tok + rows};
    }
    void prefill(const Slot& slot,const Payload& payload,uint32_t first,uint32_t rows) {
        const auto t0 = std::chrono::steady_clock::now();   // for the error-word line: a ~10 s quantum is a GPU wait timeout
        require(rows <= uint64_t(c_.prefill_rows)*(c_.two_lanes?2:1), "prefill quantum exceeds native lanes");
        c_.begin_prefill(slot); // real run/view selector supplied by TP4/fc2
        const int a = int(first), ma = std::min(int(rows),c_.prefill_rows), mb = int(rows)-ma;
        std::vector<int> pos(rows); for (unsigned r=0;r<rows;++r) pos[r]=a+int(r);
        auto& p = c_.prefill; auto& p2 = c_.prefill2; auto& dq = c_.draft_prefill;
        g6rt::pf_inputs(p,ma,payload.prompt.data()+a,pos.data());
        g6rt::pf_enqueue(c_.prefill_fabric,p,prefill_sequence_,c_.main.nl,ma);
        if (mb) {
            g6rt::pf_inputs(p2,mb,payload.prompt.data()+a+ma,pos.data()+ma);
            g6rt::pf_enqueue(c_.prefill_fabric2,p2,prefill_sequence2_,c_.main.nl,mb);
            if (c_.pf_enter) c_.pf_enter();
            if (c_.lane_streams) g6rt::pf_step2s(c_.main,p,p2,ma,mb,a+ma,a+ma+mb,prefill_sequence_,prefill_sequence2_,c_.stream,*c_.lane_streams);
            else g6rt::pf_step2(c_.main,p,p2,ma,mb,a+ma,a+ma+mb,prefill_sequence_,prefill_sequence2_,c_.stream);
            prefill_sequence2_ += c_.main.nl;
        } else {
            if (c_.pf_enter) c_.pf_enter();
            g6rt::pf_step(c_.main,p,ma,a+ma,prefill_sequence_,c_.stream);
        }
        if (c_.pf_exit) c_.pf_exit();
        fence(); prefill_sequence_ += c_.main.nl;
        g6rt::pf_draft_inputs(dq,ma,payload.prompt.data()+a+1,pos.data());
        g6rt::pf_draft(c_.draft,dq,p.x,ma,a+ma,c_.stream); fence();
        if (mb) {
            g6rt::pf_draft_inputs(dq,mb,payload.prompt.data()+a+ma+1,pos.data()+ma);
            g6rt::pf_draft(c_.draft,dq,p2.x,mb,a+ma+mb,c_.stream); fence();
        }
        {
            // Name every word before the fail-closed require, as launch() does for steps. Codes: fabric 0x61.. = a GPU wait
            // for a peer's rows timed out (glue::ERR_AR | coll << 8 | peer), 0x70../0x71../0x74.. = the proxy (it printed);
            // attention bits 1 position, 2 page, 4 NaN, 8 reuse row, 16 no end, 32 cell bounds; extra 0x63000400 = a
            // missing owner page in the CP staging fill, 0x63000800 = a staging hash mismatch, 0x63000c00 = a staging wait
            // timed out, 0x65.. = the QO pair wire (qp_wire.h)
            const unsigned f1 = *c_.prefill_fabric.dev().error, f2 = mb ? *c_.prefill_fabric2.dev().error : 0u;
            int lm = -1, ld = -1;
            const unsigned am = g6rt::attn_errors_at(c_.main, &lm), ad = g6rt::attn_errors_at(c_.draft.d, &ld);
            const unsigned ce = c_.extra_error ? c_.extra_error() : 0u;
            if (f1 || f2 || am || ad || ce)
                fprintf(stderr,"# S2 prefill error words rank %d: fabric %08x/%08x attn main %08x (layer %d) draft %08x (layer %d) extra %08x "
                        "(request %llu first %u rows %u, %.1f ms into the quantum, at %lld ms)\n",c_.main.rank,f1,f2,am,lm,ad,ld,ce,
                        (unsigned long long)slot.req.id,first,rows,
                        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),(long long)s2_wall_ms());
            require(!f1 && !f2 && !am && !ad && !ce, "S2 prefill fabric/attention error");
        }
        c_.end_prefill(); // restore decode views before yielding; NEVER late_draft
    }
public:
    explicit NativeDispatch(NativeContext c, const Config& scheduler) : c_(std::move(c)) {
        scheduler.validate();
        require(c_.main.s2 && c_.draft.d.s2 && c_.main.graph && c_.draft.d.graph && c_.split == 3 && c_.split < c_.main.nl,
                "native S2 needs initialized pools and production split3 graphs");
        require(c_.prefill_rows > 0 && c_.input && c_.draft_input && c_.output && c_.draft_output &&
                bool(c_.enqueue_main) && bool(c_.solo_main) && bool(c_.solo_draft) && bool(c_.begin_prefill) && bool(c_.end_prefill), "incomplete native context");
        require(scheduler.max_rows <= g6rt::MX && scheduler.limit() <= unsigned(c_.main.s2->nslots) &&
                scheduler.limit() <= unsigned(c_.draft.d.s2->nslots) && scheduler.pool_rows / PAGE >= unsigned(c_.main.s2->pool_pages) &&
                scheduler.pool_rows / PAGE >= unsigned(c_.draft.d.s2->pool_pages) && // logical capacity >= physical (pool.h)
                scheduler.prefill_rows == unsigned(c_.prefill_rows) &&
                scheduler.two_lanes == c_.two_lanes, "native storage/row/lane limits differ from advertised config");
        require(c_.main.in == c_.input && c_.draft.d.in == c_.draft_input && c_.main.out == c_.output && c_.draft.d.out == c_.draft_output,
                "native staging pointers differ from captured model inputs");
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&layout_),sizeof(g6rt::S2In),cudaHostAllocMapped));
        G6RT_CK(cudaHostAlloc(reinterpret_cast<void**>(&draft_layout_),sizeof(g6rt::S2In),cudaHostAllocMapped));
        *layout_ = {}; *draft_layout_ = {};
        previous_layout_ = c_.main.s2->in; previous_draft_layout_ = c_.draft.d.s2->in;
        c_.main.s2->in = layout_; c_.draft.d.s2->in = draft_layout_;
    }
    ~NativeDispatch() {
        for (auto& item:graphs_) destroy(item.second);
        c_.main.s2->in = previous_layout_; c_.draft.d.s2->in = previous_draft_layout_;
        cudaFreeHost(layout_); cudaFreeHost(draft_layout_);
    }
    NativeDispatch(const NativeDispatch&) = delete; NativeDispatch& operator=(const NativeDispatch&) = delete;
    // graphs are keyed by (kind, M, bucket), not by session: a retired session leaves them valid
    void retire(const Slot&) {}
    // Boot capture of every graph key a decode step can reach, before serving (a launch of any other key fails):
    //   main : M in 1..chunk (prompt tails, class row_class(M)) U S*(k+1), S = 1..sessions, k = 1..K with G6_S2_ADAPT
    //          (else k = K), class row_class(K+1)                          -> two graphs (split A / B)
    //   draft: M in 1..max(sessions*(K+1), chunk)                          -> first form
    //   chain: M in 1..sessions (one row per session; only if K >= 2)
    // each at every context bucket. The synthetic layout is irrelevant to the graph (S2_ATTN_LAYOUT_FREE): the sessions' rows,
    // contexts and source rows are read from S2In at launch. Call before the first launch / before connecting.
    void capture_all(unsigned sessions, int mtp, int chunk) {
        require(g6rt::S2_ATTN_LAYOUT_FREE, "boot capture needs the layout-free S2 attention: the graphs would bake the session layout");
        require(sessions >= 1 && sessions <= unsigned(g6rt::S2N) && mtp >= 1 && chunk >= 1 && chunk <= g6rt::MX, "invalid boot capture set");
        const int N = int(sessions), K = mtp, nbk = c_.main.nbk;
        require(nbk >= 1 && nbk == c_.draft.d.nbk && nbk <= g6rt::NBK, "main/draft context buckets differ");
        std::set<std::pair<int,int>> mains;   // (M, k1 class)
        std::set<int> drafts, chains;
        for (int M = 1; M <= chunk; ++M) { mains.insert({M, s2::row_class(M)}); drafts.insert(M); }
        for (int S = 1; S <= N; ++S)
            for (int k = s2::adapt_env() ? 1 : K; k <= K; ++k) mains.insert({S * (k + 1), s2::row_class(K + 1)});
        for (int M = 1; M <= N * (K + 1); ++M) drafts.insert(M);
        if (K >= 2) for (int M = 1; M <= N; ++M) chains.insert(M);
        for (const auto& mc : mains) require(mc.first <= g6rt::MX, "main step rows exceed G6_MX");
        for (int M : drafts) require(M <= g6rt::MX, "draft step rows exceed G6_MX");
        const auto t0 = std::chrono::steady_clock::now();
        size_t free0 = 0, total = 0, free1 = 0; G6RT_CK(cudaMemGetInfo(&free0,&total));
        const bool on_main = c_.main.s2on, on_draft = c_.draft.d.s2on;
        c_.main.s2on = c_.draft.d.s2on = true; fence();
        size_t keys = 0, graphs = 0;
        // one synthetic step: ns sessions of rows each (first form: one session of M rows; chain: M sessions of one row)
        auto one = [&](bool draft, bool chain, int M, int k, int cls) {
            auto& model = draft ? c_.draft.d : c_.main;
            g6rt::S2In h{};
            h.ns = chain ? M : 1; h.cls = cls;
            for (int j = 0; j < h.ns; ++j) {
                h.sess[j] = j; h.row0[j] = chain ? j : 0; h.rows[j] = chain ? 1 : M; h.ctx[j] = model.bk[k]; h.src0[j] = 0;
            }
            model.s2->h = h; model.s2->bucket = k;
            for (int j = 0; j < g6rt::S2N; ++j) model.s2->bk[j] = k;
            g6rt::s2_prepare(model,M,c_.stream); fence();
            Graphs g;
            if (draft) g.a = capture([&]{g6rt::launch_draft_step(c_.draft,M,model.bk[k],chain,c_.stream);});
            else {
                g.a = capture([&]{g6rt::launch_step(c_.main,M,model.bk[k],c_.stream,0,c_.split);});
                g.b = capture([&]{g6rt::launch_step(c_.main,M,model.bk[k],c_.stream,1,c_.split);});
            }
            const std::string key = launch_key(draft ? std::string(chain ? "c" : "d") + std::to_string(M)
                                                     : "m" + std::to_string(M) + "c" + std::to_string(cls), k);   // = Dispatch graph_key (s2::Step::key)
            graphs += g.b ? 2 : 1; ++keys;
            graphs_.emplace(key,std::move(g));
        };
        for (int k = 0; k < nbk; ++k) {
            for (const auto& mc : mains) one(false,false,mc.first,k,mc.second);
            for (int M : drafts) one(true,false,M,k,s2::row_class(M));
            for (int M : chains) one(true,true,M,k,s2::row_class(M));
        }
        c_.main.s2on = on_main; c_.draft.d.s2on = on_draft; fence();
        G6RT_CK(cudaMemGetInfo(&free1,&total));
        if (c_.main.rank == 0)
            fprintf(stderr,"# s2 boot capture: %zu keys, %zu graphs (%d buckets; main %zu draft %zu chain %zu shapes; N %d K %d chunk %d) in %.2f s, graph memory %.1f MiB\n",
                keys,graphs,nbk,mains.size(),drafts.size(),chains.size(),N,K,chunk,
                std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(),
                (double(free0)-double(free1))/(1024.0*1024.0));
    }
    DispatchHooks hooks() {
        return {
            [&](bool draft,int ctx){return g6rt::bucket_of(draft?c_.draft.d:c_.main,ctx);},
            [&](int old,int next){mode(old,next);},
            [&](const Launch& l){return launch(l);},
            [&](const Slot& s,const Payload& p,uint32_t first,uint32_t rows){prefill(s,p,first,rows);},
            [&](int row,bool target,int k){
                require(row >= 0 && row < lp_rows_, "logprob row outside the last main step");
                return g6rt::lp_row(lp_,row,target,k);
            }
        };
    }
    void synchronize() { fence(); }
};
} // namespace g6s2
#endif // DEC_S2
