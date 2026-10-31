// DSpark round runner for the DS41 serve (B12): the Step::DRAFT / Step::CHAIN branch of serve_ds41's launch hook (s2/ds41_native.cuh).
// No s2_batch.h change: Batch's MTP flow (sc.mtp = k) already runs MAIN verify (k + 1 rows) -> ONE DRAFT step -> CHAIN steps j = 1..k-1.
// DSpark drafts all 5 tokens in one block pass, so the DRAFT step runs the block and the CHAIN steps only hand out the cached drafts:
//   DRAFT  rows are the accepted rows of the verify (or a prompt chunk): Row{tok = next token, pos}. The last row is the round's end: q = its pos
//          (last accepted position), golden = its tok (the token at q + 1), accepted = M - 1 (the row index of the verify chunk). The host
//          knows all three, so the device accept is bypassed (Dsp::host_accept; the device rule is wrong for a prompt tail of M > 1 rows).
//          u[M - 1] = draft[0] = d[1]; the other rows return 0 (Batch reads only the last).
//   CHAIN  one row, pos = q + j, tok = d[j] = draft[j - 1]; returns d[j + 1] = draft[j]. Nothing is launched.
//   A prompt chunk's first-form draft (Batch::prompt: round id -1, event ROUND) is skipped: the main steps fill the draft layers' key rings
//   (ST_DS_MAIN), the block only runs for the last chunk (event TAIL) and for rounds / forced chunks (round id >= 0).
// After the accept and before the next main step the main Model's compressor state needs ds41_attn_comp_commit_all (B12-comp-commit.patch):
// the round calls it through Ds41Dspark::comp_commit.
// The host rules (q / golden / acc, the prompt-draft skip, what DRAFT and CHAIN return, the fail-closed checks) are in ds41_dspark_round_logic.h,
// which a CPU test runs against the real s2::Batch (tests/batch_ds41_dspark_test.cpp); this file adds the CUDA parts.
// SESS > 1 (Model::ds_slots = S2 slots > 1): one Ds41RoundLogic per slot. A DRAFT step holds several sessions of acc + 1 rows each (not uniform):
// session i of the block (the drafted ones, in step order) gets q / golden / acc [i], crow0[i] = its first row in the last MAIN step
// (Launch::source_rows), cslot[i] = its slot; the draft StepIn has 5 rows per session with si.sess = slot, the draft Model's ds_ns = sessions.
// The compressor commit is ds41_attn_comp_commit_sessions; a CHAIN step is one row per session, each from its slot's cached drafts.
#pragma once
#if G6_MODEL_DS41
#include "service.h"
#include "ds41_dspark_round_logic.h"
#include <algorithm>
#include <cstring>
namespace g6s2 {

// launch_step_ds41's units for the DSpark draft Model (plan_draft): the plan has no U_STEP_OUT (the heads write dsp.draft / dsp.conf), so the
// step-out kernel is launched here for the attention error word.
inline void launch_step_ds41_draft(g6rt::Model& m, int M, cudaStream_t st) {
    using namespace g6rt;
    ds41_unit(m, ds41::U_STEP_IN, 0, M, st);     // ST_DS_DRAFT_IN: step cells, accept (bypassed by host_accept), block tokens / positions, embed
    ds41_unit(m, ds41::U_EMBED, 0, M, st);       // hc_attn of layer 0
    for (int i = 0; i < m.nl; ++i) {
        ds41_unit(m, ds41::U_ATTN, i, M, st);
        ds41_unit(m, ds41::U_CONS_A, i, M, st);
        ds41_unit(m, ds41::U_MLP, i, M, st);
        ds41_unit(m, ds41::U_CONS_F, i, M, st);
    }
    ds41_unit(m, ds41::U_HEAD_LOCAL, 0, M, st);
    ds41_unit(m, ds41::U_HEAD_MERGE, 0, M, st);  // ST_DS_MARKOV: 5 merges (collectives 6..10), confidence, dsp.draft
    launch_step_out(m, M, st);                   // out->attn_err (the sampled ids of the draft rows are not used)
}

// what g6m4.cu hands the runner (the draft Model, its fabric / graph / eager launch, and the main Model's compressor commit)
struct Ds41Dspark {
    g6rt::Model& draft;
    g6rt::StepIn* input;                                   // the draft Model's mapped StepIn (draft.in)
    const g6rt::StepOut* output;                           // the draft Model's mapped StepOut (draft.out)
    std::function<void(const g6rt::StepIn&)> enqueue;      // the draft fabric: this step's 11 collective sizes (6 layer exchanges, 5 head merges)
    std::function<void()> launch;                          // the captured draft graph of draft.ds_ns sessions, or launch_step_ds41_draft eagerly (no --graph)
    std::function<unsigned()> errors;                      // the draft fabric's error word (0 = clean), read after the stream is synchronized
    std::function<void()> comp_commit;                     // ds41_attn_comp_commit_all(&main, dsp.q, dsp.acc, stream); SESS > 1: _sessions(.., draft.ds_ns)
};

class Ds41Round {
public:
    static constexpr int BLOCK = Ds41RoundLogic::BLOCK;
    Ds41Round(g6rt::Model& main, Ds41Dspark& d, cudaStream_t st, int k)
        : main_(main), d_(d), st_(st), slots_(main.ds_slots), logic_(size_t(std::max(1, main.ds_slots)), Ds41RoundLogic(k)), conf_(size_t(std::max(1, main.ds_slots)) * BLOCK) {
        const auto& D = d.draft.dsp;
        const int dsp_nsess = std::min(slots_, ::ds41dsp::MARKOV_MAXS);
        require(k >= 1 && k <= BLOCK && main.dsp.k == k && slots_ >= 1 && main.dsp.nsess == dsp_nsess && D.nsess == dsp_nsess && d.draft.ds_slots == slots_ &&
                D.draft_model && D.host_accept && D.q && D.golden && D.acc && (slots_ == 1 || (D.crow0 && D.cslot && dsp_nsess * BLOCK <= g6mx::MX)) &&
                D.draft && D.conf && d.input && d.output && d.enqueue && d.launch && d.errors && d.comp_commit,
                "DSpark round: the draft Model, its fabric / graph or the compressor commit is not wired (host_accept, comp_commit, k, slots)");
        G6RT_CK(cudaHostAlloc((void**)&hin_, 5 * size_t(slots_) * sizeof(int), cudaHostAllocDefault));
        G6RT_CK(cudaHostAlloc((void**)&hout_, 2 * size_t(slots_) * BLOCK * sizeof(int), cudaHostAllocDefault));
    }
    Ds41Round(const Ds41Round&) = delete;
    Ds41Round& operator=(const Ds41Round&) = delete;
    ~Ds41Round() { cudaFreeHost(hin_); cudaFreeHost(hout_); }

    // a new request at position 0: the cached drafts are stale and the draft layers' key rings start empty (the draft step counter is NOT reset:
    // the draft fabric's steps stay consecutive)
    void reset() {
        logic_[0].reset();
        g6rt::reset(d_.draft, st_);
    }
    // SESS > 1: a fresh run on slot q: that slot's drafts are stale and its slab of the draft layers' key rings is cleared (no step in flight)
    void reset(int q) {
        require(slots_ > 1 && q >= 0 && q < slots_, "DSpark round: slot reset outside the S2 slots");
        logic_[size_t(q)].reset();
        require(g6rt::ds41_attn_clear(&d_.draft, st_, q) == 0, "DSpark round: draft ring slot clear failed");
    }

    // Step::DRAFT
    std::vector<int> draft_step(const Launch& l) {
        if (slots_ > 1) return draft_step_sessions(l);
        require(l.solo == 0, "DSpark draft step is not solo");
        Ds41RoundLogic& logic_ = this->logic_[0];
        Ds41RoundLogic::DraftIn in;
        if (!logic_.draft_in(l.step, in)) return logic_.skipped(l.step);   // a prompt chunk before the last: nothing to draft from yet
        const auto& D = d_.draft.dsp;
        hin_[0] = in.q; hin_[1] = in.golden; hin_[2] = in.acc;
        G6RT_CK(cudaMemcpyAsync(D.q, hin_ + 0, sizeof(int), cudaMemcpyHostToDevice, st_));
        G6RT_CK(cudaMemcpyAsync(D.golden, hin_ + 1, sizeof(int), cudaMemcpyHostToDevice, st_));
        G6RT_CK(cudaMemcpyAsync(D.acc, hin_ + 2, sizeof(int), cudaMemcpyHostToDevice, st_));
        g6rt::StepIn si{};
        si.step = ++dseq_;
        si.M = BLOCK;
        si.context = in.q + 1 + BLOCK;
        *d_.input = si;
        d_.enqueue(si);
        d_.comp_commit();                          // the main Model's compressor state of the accepted row (needs dsp.q / dsp.acc: written above)
        d_.launch();
        G6RT_CK(cudaMemcpyAsync(hout_, D.draft, BLOCK * sizeof(int), cudaMemcpyDeviceToHost, st_));
        G6RT_CK(cudaMemcpyAsync(hout_ + BLOCK, D.conf, BLOCK * sizeof(float), cudaMemcpyDeviceToHost, st_));
        G6RT_CK(cudaStreamSynchronize(st_));
        if (const unsigned e = d_.errors()) require(false, draft_fabric_error_text(e).c_str());   // names the stuck collective and peer
        require(!d_.output->attn_err, "DSpark draft step attention error");
        std::memcpy(conf_.data(), hout_ + BLOCK, BLOCK * sizeof(float));
        return logic_.draft_out(l.step, in.q, hout_);
    }

    // Step::CHAIN
    std::vector<int> chain_step(const Launch& l) {
        if (slots_ > 1) {   // one row per session, each from its slot's cached drafts
            const s2::Step& s = l.step;
            require(s.kind == s2::Step::CHAIN && s.rows.size() == s.sess.size() && s.row0.size() == s.sess.size(), "DSpark chain step layout");
            std::vector<int> u(size_t(s.M()), 0);
            for (size_t j = 0; j < s.sess.size(); ++j) {
                const int q = s.sess[j];
                require(q >= 0 && q < slots_ && s.rows[j] == 1, "DSpark chain step: one row per session on an S2 slot");
                u[size_t(s.row0[j])] = logic_[size_t(q)].chain_row(s.r[size_t(s.row0[j])]);
            }
            return u;
        }
        require(l.solo == 0, "DSpark chain step is not solo");
        return logic_[0].chain(l.step);
    }

    float confidence(int i, int slot = 0) const { return conf_[size_t(slot) * BLOCK + size_t(i)]; }   // raw FP32 score of draft i (diagnostics)

private:
    // SESS > 1: Step::DRAFT of 1..slots sessions (the drafted ones form the block, in step order)
    std::vector<int> draft_step_sessions(const Launch& l) {
        const s2::Step& s = l.step;
        require(l.source_rows.size() == s.sess.size(), "DSpark draft step: no source rows");
        std::vector<int> u(size_t(s.M()), 0);
        std::vector<size_t> js;
        std::vector<Ds41RoundLogic::DraftIn> ins;
        for (size_t j = 0; j < s.sess.size(); ++j) {
            const int q = s.sess[j];
            require(q >= 0 && q < slots_, "DSpark draft step session outside the S2 slots");
            Ds41RoundLogic::DraftIn in;
            if (logic_[size_t(q)].draft_in_sess(s, j, in)) { js.push_back(j); ins.push_back(in); }
        }
        const int S = int(js.size());
        if (!S) return u;                          // prompt chunks before the last: nothing to draft from yet
        const auto& D = d_.draft.dsp;
        require(S <= D.nsess, "DSpark draft step sessions exceed DSpark nsess");
        g6rt::StepIn si{};
        si.step = ++dseq_;
        si.M = S * BLOCK;
        for (int i = 0; i < S; ++i) {
            const int q = s.sess[js[size_t(i)]];
            hin_[i] = ins[size_t(i)].q; hin_[S + i] = ins[size_t(i)].golden; hin_[2 * S + i] = ins[size_t(i)].acc;
            hin_[3 * S + i] = l.source_rows[js[size_t(i)]]; hin_[4 * S + i] = q;
            for (int r = 0; r < BLOCK; ++r) si.sess[i * BLOCK + r] = q;
            si.context = std::max(si.context, ins[size_t(i)].q + 1 + BLOCK);
        }
        int* dst[5] = {D.q, D.golden, D.acc, D.crow0, D.cslot};
        for (int a = 0; a < 5; ++a) G6RT_CK(cudaMemcpyAsync(dst[a], hin_ + a * S, size_t(S) * sizeof(int), cudaMemcpyHostToDevice, st_));
        d_.draft.ds_ns = S;                        // the ops, the graph choice and the compressor commit read it
        *d_.input = si;
        d_.enqueue(si);
        d_.comp_commit();                          // the main Model's compressor state of each session's accepted row (q / acc / crow0 / cslot: above)
        d_.launch();
        G6RT_CK(cudaMemcpyAsync(hout_, D.draft, size_t(S) * BLOCK * sizeof(int), cudaMemcpyDeviceToHost, st_));
        G6RT_CK(cudaMemcpyAsync(hout_ + size_t(slots_) * BLOCK, D.conf, size_t(S) * BLOCK * sizeof(float), cudaMemcpyDeviceToHost, st_));
        G6RT_CK(cudaStreamSynchronize(st_));
        d_.draft.ds_ns = 1;
        if (const unsigned e = d_.errors()) require(false, draft_fabric_error_text(e).c_str());
        require(!d_.output->attn_err, "DSpark draft step attention error");
        for (int i = 0; i < S; ++i) {
            const size_t j = js[size_t(i)];
            const int q = s.sess[j];
            std::memcpy(&conf_[size_t(q) * BLOCK], hout_ + size_t(slots_) * BLOCK + size_t(i) * BLOCK, BLOCK * sizeof(float));
            u[size_t(s.row0[j] + s.rows[j] - 1)] = logic_[size_t(q)].draft_set(ins[size_t(i)].q, hout_ + size_t(i) * BLOCK);
        }
        return u;
    }

    g6rt::Model& main_;
    Ds41Dspark& d_;
    cudaStream_t st_;
    int slots_;
    std::vector<Ds41RoundLogic> logic_;            // one per S2 slot (slots_ == 1: logic_[0], the one-session path)
    int* hin_ = nullptr;
    int* hout_ = nullptr;
    std::vector<float> conf_;
    uint64_t dseq_ = 0;
};
}
#endif
