// The S2 runtime signature: the string every rank sends to the bootstrap agreement (s2/bootstrap.h), so ranks whose
// options, knobs or loaded files would plan differently refuse to start instead of diverging. One format per model.
// SigInputs holds what each format reads, including values main has only after the load (the checkpoint store's
// blocks, the vision tower's id, the QO table size) and the strings other builders make (knob::signature(),
// s2::ngram_sig, s2::kstopb_sig, s2::DSelTable::str). Host only.
#pragma once
#include <cstdint>
#include <string>
#include "../ds41_vision_id.h"

namespace g6s2
{

enum class SigModel { Glm, Flash, Ds41 };

struct SigInputs
{
    SigModel model = SigModel::Glm;
    int cap = 0;
    std::string layers_s;               // --layers as given
    int chunk = 0, nc = 0;              // Plan::chunk, --top-p-cands
    std::string set_id;
    bool dfl = false;                   // --dflash-manifest
    int dfl_k = 0, s2sess = 1;
    std::string knobs;                  // knob::signature(): the agree-scope knobs off their defaults
    int MP = 0, mtp = 0;                // GLM and Flash
    int dfl_reuse = 0, dfl_reuse_min = 0;
    int lanes = 0, gsplit = 0;          // Flash: prefill lanes, GR8_DEC_SPLIT
    std::string ckpt;                   // Flash: g6s2::flash_cache_name, empty when GR8_GLM53F_CACHE is off
    uint64_t ckpt_blocks = 0;           // the checkpoint store's blocks
    bool ckpt_persist = false;
    int s2_below = 0;                   // GLM
    bool two = false;                   // two prefill lanes
    int pf_own = 0;
    bool pf_qp = false;                 // GR8_PF_OWN=4
    int qp_ctx = 0, qp_min = 0;         // the QO table size (Prefill::qp_ctx), pf_qp_min(): read with pf_qp only
    bool grammar = false;               // g6rt::grammar_ready()
    std::string ngram;                  // s2::ngram_sig
    int kstop_mode = 0, kstop_kmax = 0;
    double kstop_tau[4] = {};
    double kstop_seed3 = 0;
    int draft_par = 0, qs_mode = 0;
    bool iq_split = false, mtp_split = false;
    int dhead_on = 0, dhead_tpc = 0, dhead_ks = 0;   // DraftHeadI4
    bool mtp_last = false;
    int dhead2 = 0;
    bool dfl_i4 = false, dfl_body_i4 = false;
    float dfl_alpha = 1.f;
    int dsel_model = 0;
    std::string dsel_tab;               // s2::DSelTable::str
    std::string kstopb;                 // s2::kstopb_sig
    int cp_step = 0;
    bool cp_seg = false;
    int ctx_sp = 0, posidx = 0;         // g6_attn_ctx_sp(), g6_attn_posidx()
    bool steer_shared = false;          // a loaded steering profile of shared scope: the S2 layers carry its MoE finalize
    int ds_mtp = 0;                     // DS41: GR8_DS_MTP
    bool sh_tc = false;                 // GR8_DS_SH_TC (W8A8 shared expert, not bit-identical to off)
    int l2hint = 0, fc2stg = 0, fc1bstg = 0;
    bool c8_pack = false, sess_drafts = false;
    int pf_lanes = 1;
    int tk_main = 0, tk_decode = 0;     // GR8_DS_MOE_TOPK and its decode cut
    int idx_pipe_b = 0, idx_q_trig = 0, idx_pipe_qr = 0;
    bool reuse = false;
    bool vis_on = false;                // the vision tower loaded (the capability, not the knob)
    std::string vis_id;                 // ds41v::file_id of the tower file
};

inline std::string runtime_signature(const SigInputs& in)
{
    if (in.model == SigModel::Flash)
    {
        std::string s = "flash solo cap " + std::to_string(in.cap) + " layers " + in.layers_s +
            " mp " + std::to_string(in.MP) + " lanes " + std::to_string(in.lanes) + " mtp " + std::to_string(in.mtp) +
            " chunk " + std::to_string(in.chunk) + " cands " + std::to_string(in.nc) + " set " + in.set_id +
            (in.dfl ? " dflash k " + std::to_string(in.dfl_k) : std::string()) +
            (in.dfl_reuse ? " dflreuse " + std::to_string(in.dfl_reuse) + "/" + std::to_string(in.dfl_reuse_min) : std::string()) +
            (in.s2sess > 1 ? " sessions " + std::to_string(in.s2sess) + " split " + std::to_string(in.gsplit) : std::string()) +
            in.knobs;
        // every rank must keep the same checkpoint entries: the mode, the store's blocks and whether they persist
        if (!in.ckpt.empty())
            s += " ckpt " + in.ckpt + " blocks " + std::to_string(in.ckpt_blocks) + (in.ckpt_persist ? " persist" : "");
        return s;
    }
    if (in.model == SigModel::Ds41)
    {
        std::string s = std::string(in.s2sess > 1 ? "ds41 sessions " + std::to_string(in.s2sess) : std::string("ds41 solo")) +
            " cap " + std::to_string(in.cap) + " layers " + in.layers_s + " mtp " + std::to_string(in.ds_mtp) +
            " chunk " + std::to_string(in.chunk) + " cands " + std::to_string(in.nc) + " set " + in.set_id +
            (in.dfl ? " dflash k " + std::to_string(in.dfl_k) : std::string()) +
            " shtc " + std::to_string(in.sh_tc ? 1 : 0) +
            (in.l2hint ? " l2h " + std::to_string(in.l2hint) : std::string()) +
            (in.fc2stg ? std::string(" stg 1") : std::string()) +
            (in.fc1bstg ? std::string(" fc1bstg 1") : std::string());
        if (in.c8_pack) s += " c8pack 1";
        if (in.sess_drafts) s += " sessdrafts 1";
        if (in.pf_lanes == 2) s += " pflanes 2";
        if (in.tk_decode != in.tk_main) s += " tkmain " + std::to_string(in.tk_main) + " tkdecode " + std::to_string(in.tk_decode);
        if (in.idx_pipe_b) s += " idxpipeb 1";
        if (in.idx_q_trig) s += " idxqtrig 1";
        if (in.idx_pipe_qr) s += " idxpipeqr 1";
        if (in.reuse) s += " reuse 1";
        // with the tower, " vision <file id>" (header and size agreement, not tensor data); a rank without the file
        // boots text-only and refuses beside ranks with it
        s += ds41v::signature(in.vis_on, in.vis_id);
        return s + in.knobs;
    }
    return "g6m4 s2 below " + std::to_string(in.s2_below) + " cap " + std::to_string(in.cap) + " layers " + in.layers_s +
        " mp " + std::to_string(in.MP) + " lanes " + std::to_string(in.two ? 2 : 1) + " mtp " + std::to_string(in.mtp) + " chunk " +
        std::to_string(in.chunk) + " cands " + std::to_string(in.nc) + " split 3 own " + std::to_string(in.pf_own) +
        " qpctx " + std::to_string(in.pf_qp ? in.qp_ctx : 0) + " qpmin " + std::to_string(in.pf_qp ? in.qp_min : 0) +
        (in.set_id.empty() ? "" : " set " + in.set_id) + (in.grammar ? " grammar" : "") +
        (in.dfl_reuse ? " dflreuse " + std::to_string(in.dfl_reuse) + "/" + std::to_string(in.dfl_reuse_min) : "") + in.ngram +
        (in.kstop_mode ? " kstop " + std::to_string(in.kstop_mode) + " kmax " + std::to_string(in.kstop_kmax) + " tau " +
                         std::to_string(in.kstop_tau[1]) + "," + std::to_string(in.kstop_tau[2]) + "," + std::to_string(in.kstop_tau[3]) +
                         " seed3 " + std::to_string(in.kstop_seed3) : "") +
        (in.draft_par ? " draftpar " + std::to_string(in.draft_par) : "") +
        (in.qs_mode ? " qasplit " + std::to_string(in.qs_mode) : "") +
        (in.iq_split ? " idxqsplit 1" : "") + (in.mtp_split ? " mtpsplit 1" : "") +
        (in.dhead_on ? " dheadi4 " + std::to_string(in.dhead_tpc) + "x" + std::to_string(in.dhead_ks) : "") +
        (in.mtp_last ? " mtplast 1" : "") + (in.dhead2 ? " dhead2 " + std::to_string(in.dhead2) : "") +
        (in.dfl_i4 ? " dflheadi4 " + std::to_string(in.dhead_tpc) + "x" + std::to_string(in.dhead_ks) : "") +
        (in.dfl_body_i4 ? " dflbodyi4 1" : "") +
        (in.dfl_alpha != 1.f ? " dflalpha " + std::to_string(in.dfl_alpha) : "") +
        (in.dsel_model ? " dselmodel " + std::to_string(in.dsel_model) + " " + in.dsel_tab : "") + in.kstopb +
        (in.cp_step ? " cpsr " + std::to_string(in.cp_step) + (in.cp_seg ? " seg" : "") : "") +
        (in.ctx_sp ? " ctxsp " + std::to_string(in.ctx_sp) : "") + (in.posidx ? " posidx 1" : "") +
        (in.steer_shared ? " s2steer 1" : "") +
        in.knobs;
}

}   // namespace g6s2
