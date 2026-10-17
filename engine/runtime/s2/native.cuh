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
// G6_S2_TIMELINE=N (every rank, N > 0): where the time between decode steps goes. Each step launch (main, first-form
// draft, chain here; the DFlash2 drafter steps in g6m4) stamps the host clock on entry, before and after its graph launch
// and after its fence, and records a CUDA event before and after its graph(s). After the fence it adds the step to its
// kind's totals (GPU span = event to event; host input prep, launch call, fence wait) and the transition from the step
// before to that pair's totals (the main-stream gap = the previous step's end event to this step's start event, both on
// the main stream; host = the previous
// fence to this launch's entry: accept, emit, drafter choice, step planning). Prints them every N main steps. The gap is
// bounded by the main stream's events: with G6_DRAFT_PAR the drafter's side stream can run inside it, so it is an upper
// bound on what launching the next step from the GPU could remove; the spans stay. Unset: no events, no stamps.
// G6_DRAFT_PAR: a solo DFlash2 round's drafter step (launched before the first form on its own stream, waited for after
// the first form and the chain) counts as dfl_ctx_par / dfl_query_par through par_*, with its own event pair: span = its
// GPU start to end on its stream (under the first form and chain, so it overlaps their spans), prep and launch as above,
// wait = the phase 2 wait alone (the drafter's tail the first form and chain did not hide). It is not in the
// transitions, which stay on the target stream: main>draft's host time holds the phase 1 launch, and the idle before
// the next main step holds that tail.
struct S2Timeline {
    enum Kind { MAIN, DRAFT, CHAIN, DCTX, DQUERY, DCTX_PAR, DQUERY_PAR, NK };
    using clk = std::chrono::steady_clock;
    int every = 0;
    cudaEvent_t ev[2][2] = {};          // [pair][start, end]; pairs alternate, so the previous step's end stays readable
    int cur = 0, prev_kind = 0;
    bool have_prev = false;
    clk::time_point t_enter, t_pre, t_post, t_prev;
    cudaEvent_t evp[2] = {};            // G6_DRAFT_PAR: the drafter step on its own stream [start, end]
    clk::time_point p_enter, p_pre, p_post, p_wait;
    long n[NK] = {}, rows[NK] = {}, tn[NK][NK] = {}, mains = 0, breaks = 0;
    double span[NK] = {}, prep[NK] = {}, call[NK] = {}, wait[NK] = {}, idle[NK][NK] = {}, host[NK][NK] = {};
    static const char* name(int k) { static const char* s[NK] = {"main", "draft", "chain", "dfl_ctx", "dfl_query", "dfl_ctx_par", "dfl_query_par"}; return s[k]; }
    static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double,std::milli>(b - a).count(); }
    S2Timeline() {
        const char* e = getenv("G6_S2_TIMELINE");
        every = e ? atoi(e) : 0;
        if (every > 0) for (auto& p : ev) for (auto& x : p) G6RT_CK(cudaEventCreate(&x));
        if (every > 0) for (auto& x : evp) G6RT_CK(cudaEventCreate(&x));
    }
    bool on() const { return every > 0; }
    void enter() { if (on()) t_enter = clk::now(); }
    void pre(cudaStream_t st) { if (!on()) return; t_pre = clk::now(); G6RT_CK(cudaEventRecord(ev[cur][0], st)); }
    void post(cudaStream_t st) { if (!on()) return; G6RT_CK(cudaEventRecord(ev[cur][1], st)); t_post = clk::now(); }
    // G6_DRAFT_PAR's drafter step: entry, around its launch on its stream (phase 1), phase 2's entry, after its wait
    void par_enter() { if (on()) p_enter = clk::now(); }
    void par_pre(cudaStream_t st) { if (!on()) return; p_pre = clk::now(); G6RT_CK(cudaEventRecord(evp[0], st)); }
    void par_post(cudaStream_t st) { if (!on()) return; G6RT_CK(cudaEventRecord(evp[1], st)); p_post = clk::now(); }
    void par_wait() { if (on()) p_wait = clk::now(); }
    void par_done(int kind, int M) {
        if (!on()) return;
        const auto t = clk::now();
        float s = 0; G6RT_CK(cudaEventElapsedTime(&s, evp[0], evp[1]));
        ++n[kind]; rows[kind] += M; span[kind] += s; prep[kind] += ms(p_enter, p_pre); call[kind] += ms(p_pre, p_post); wait[kind] += ms(p_wait, t);
    }
    // not a decode transition: a prefill quantum, a mode switch, the drafter's prompt fill
    void cut() { have_prev = false; }
    // after the step's fence
    void done(int kind, int M, int rank) {
        if (!on()) return;
        const auto t = clk::now();
        float s = 0; G6RT_CK(cudaEventElapsedTime(&s, ev[cur][0], ev[cur][1]));
        ++n[kind]; rows[kind] += M; span[kind] += s; prep[kind] += ms(t_enter, t_pre); call[kind] += ms(t_pre, t_post); wait[kind] += ms(t_post, t);
        if (have_prev) {
            // the between-steps gap is bounded by events on the MAIN stream only: with G6_DRAFT_PAR the drafter's
            // side stream can run inside it, so idle[] is a main-stream gap, not necessarily GPU idle time
            float g = 0; G6RT_CK(cudaEventElapsedTime(&g, ev[cur ^ 1][1], ev[cur][0]));
            if (g > 20.f) ++breaks;      // an event fence or an idle engine in between, not a round's transition
            else { ++tn[prev_kind][kind]; idle[prev_kind][kind] += g; host[prev_kind][kind] += ms(t_prev, t_enter); }
        }
        have_prev = true; prev_kind = kind; t_prev = t; cur ^= 1;
        if (kind == MAIN && ++mains % every == 0) print(rank);
    }
    void print(int rank) const {
        double ti = 0, th = 0, ds = 0;
        for (int a = 0; a < NK; ++a) for (int b = 0; b < NK; ++b) { ti += idle[a][b]; th += host[a][b]; }
        for (int k = 1; k < NK; ++k) ds += span[k];
        const double m = double(mains);
        fprintf(stderr, "# s2_tl rank %d mains %ld: per main step: main-stream gap %.3f ms (host between steps %.3f; with G6_DRAFT_PAR the drafter's side stream may run inside the gap), drafting GPU span %.3f ms, breaks %ld\n",
                rank, mains, ti / m, th / m, ds / m, breaks);
        for (int k = 0; k < NK; ++k)
            if (n[k])
                fprintf(stderr, "# s2_tl rank %d step %s n %ld rows %.2f span %.3f prep %.3f launch %.3f wait %.3f ms\n", rank, name(k), n[k],
                        double(rows[k]) / n[k], span[k] / n[k], prep[k] / n[k], call[k] / n[k], wait[k] / n[k]);
        for (int a = 0; a < NK; ++a)
            for (int b = 0; b < NK; ++b)
                if (tn[a][b])
                    fprintf(stderr, "# s2_tl rank %d trans %s>%s n %ld gap %.3f host %.3f ms\n", rank, name(a), name(b), tn[a][b],
                            idle[a][b] / tn[a][b], host[a][b] / tn[a][b]);
    }
};
inline S2Timeline& s2_tl() { static S2Timeline t; return t; }
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
    // optional: a draft step's extra fabric work after its draft fabric enqueue (G6_MTP_SPLIT: the draft gathers)
    std::function<void(unsigned long long step, int rows)> enqueue_draft_extra;
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
    float dlp_[g6rt::MX] = {};   // the last draft or chain step's StepOut::lp (dlp_rows_ rows; s2_batch.h Batch::kstop)
    int dlp_rows_ = 0;
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
        const auto t0 = std::chrono::steady_clock::now(); fence(); s2_tl().cut();
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
        S2Timeline& tl = s2_tl(); tl.enter();
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
            if (c_.enqueue_draft_extra) c_.enqueue_draft_extra(input.step,rows);
        } else c_.enqueue_main(input); // preserves existing RS order/fabric layout
        const auto tl0 = tmr_ || rlog_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}; // after fabric enqueue, BEFORE the solo/S2 launch branches (design r4 fix 2)
        tl.pre(c_.stream);
        if (l.solo >= 0) {
            if (draft) c_.solo_draft(l.buckets[0],chain,rows); else c_.solo_main(l.buckets[0],rows);
        } else {
            const auto& g = graphs_.at(l.graph_key); G6RT_CK(cudaGraphLaunch(g.a,c_.stream));
            if (g.b) G6RT_CK(cudaGraphLaunch(g.b,c_.stream));
        }
        tl.post(c_.stream);
        fence(); tl.done(draft ? (chain ? S2Timeline::CHAIN : S2Timeline::DRAFT) : S2Timeline::MAIN, rows, c_.main.rank);
        const auto out = *(draft ? c_.draft_output : c_.output);
        if (draft) { std::copy(out.lp, out.lp + rows, dlp_); dlp_rows_ = rows; }
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
        s2_tl().cut();
        require(rows <= uint64_t(c_.prefill_rows)*(c_.two_lanes?2:1), "prefill quantum exceeds native lanes");
        c_.begin_prefill(slot); // real run/view selector supplied by TP4/fc2
        const int a = int(first), ma = std::min(int(rows),c_.prefill_rows), mb = int(rows)-ma;
        std::vector<int> pos(rows); for (unsigned r=0;r<rows;++r) pos[r]=a+int(r);
        auto& p = c_.prefill; auto& p2 = c_.prefill2; auto& dq = c_.draft_prefill;
        const float scale = payload.steering.enabled ? g6rt::steering::bits_float(payload.steering.scale_bits) : 0.f;
        p.steering_scale = p2.steering_scale = scale;
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
        DispatchHooks h{
            [&](bool draft,int ctx){return g6rt::bucket_of(draft?c_.draft.d:c_.main,ctx);},
            [&](int old,int next){mode(old,next);},
            [&](const Launch& l){return launch(l);},
            [&](const Slot& s,const Payload& p,uint32_t first,uint32_t rows){prefill(s,p,first,rows);},
            [&](int row,bool target,int k){
                require(row >= 0 && row < lp_rows_, "logprob row outside the last main step");
                return g6rt::lp_row(lp_,row,target,k);
            }
        };
        h.draft_lp = [&](int row) {
            require(row >= 0 && row < dlp_rows_, "draft logprob row outside the last draft step");
            return dlp_[row];
        };
        return h;
    }
    void synchronize() { fence(); }
};
} // namespace g6s2
#endif // DEC_S2
